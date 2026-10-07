"""
predict_cascade.py — 级联推理: YOLO 缺陷检测 + HRNet 板面分割

流程 (逐张):
  1. HRNet 对整图做板面分割            -> 板面 mask
  2. YOLO 检测缺陷框                   -> 类别 + 置信度 + 坐标
  3. 融合: 计算每个框内"板面像素占比"
       - 占比 >= --min-panel-ratio   -> 判定为板上缺陷
       - 占比 <  --min-panel-ratio   -> 标记为疑似背景误检
         (默认仅标记; 加 --drop-background 则直接丢弃该框)
  4. 输出: 可视化 (原图 + 板面叠加 + 框标注) 与 summary.csv

用法示例:
    python seg/predict_cascade.py \
        --det-weights runs/pv_detect/train/weights/best.pt \
        --seg-weights runs/pv_seg/train/best.pt \
        --source /kaggle/working/solar-fault-v5/test/images \
        --limit 20

说明:
    - 分割输入尺寸默认从权重记录的 imgsz 读取 (即训练时尺寸), 也可用 --imgsz-seg 覆盖
    - 本脚本是级联管线第一步: 用板面 mask 过滤背景误检;
      缺陷像素级分割 (HRNet-2) 接入后可进一步输出缺陷面积占比
    - 板面判据: 非背景像素即视为板面 (板面分割为二值任务)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 同目录模型定义 (先加路径, 支持从任意工作目录运行)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrnet import build_hrnet  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PANEL_COLOR = (52, 199, 89)      # 板面叠加色 (绿)
RED = (255, 59, 48)              # 板上缺陷框 (红)
GRAY = (142, 142, 147)           # 疑似背景误检框 (灰)


def parse_args():
    p = argparse.ArgumentParser(description="级联推理: YOLO 缺陷检测 + HRNet 板面分割")
    p.add_argument("--det-weights", required=True, help="YOLO 检测权重 (train.py 的 best.pt)")
    p.add_argument("--seg-weights", required=True, help="HRNet 板面分割权重 (train_seg.py 的 best.pt)")
    p.add_argument("--source", required=True, help="图像文件或目录")
    p.add_argument("--out", default="runs/cascade/predict", help="输出目录")
    p.add_argument("--imgsz-det", type=int, default=640, help="YOLO 推理尺寸")
    p.add_argument("--imgsz-seg", type=int, default=0, help="HRNet 推理尺寸 (0=读权重中记录的训练尺寸)")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度阈值")
    p.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU 阈值")
    p.add_argument("--min-panel-ratio", type=float, default=0.5, help="框内板面占比判定阈值")
    p.add_argument("--drop-background", action="store_true", help="直接丢弃疑似背景误检框 (默认仅标记)")
    p.add_argument("--limit", type=int, default=0, help="最多处理多少张 (0=全部)")
    p.add_argument("--device", default="auto", help="auto / 0 / cpu / mps")
    return p.parse_args()


def pick_device(arg: str):
    """自动选择推理设备 (与 train.py 同一策略)"""
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def collect_images(source: Path, limit: int = 0) -> list:
    if source.is_file():
        imgs = [source]
    elif source.is_dir():
        imgs = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)
    else:
        raise SystemExit(f"错误: 找不到 {source}")
    if not imgs:
        raise SystemExit(f"错误: {source} 下没有图像")
    return imgs[:limit] if limit > 0 else imgs


def load_seg_model(weights: Path, device, imgsz_override: int):
    """加载 HRNet 板面分割权重 (架构/类别/尺寸从 checkpoint 读回)"""
    if not weights.exists():
        raise SystemExit(f"错误: 找不到分割权重 {weights}")
    ckpt = torch.load(weights, map_location="cpu")
    arch = ckpt.get("arch", "w18")
    num_classes = ckpt.get("num_classes", 2)
    classes = ckpt.get("classes", {"0": "background", "1": "foreground"})
    model = build_hrnet(arch, num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    imgsz = imgsz_override if imgsz_override > 0 else int(ckpt.get("imgsz", 512))
    return model, classes, arch, num_classes, imgsz


@torch.no_grad()
def segment_panel(model, img: Image.Image, imgsz: int, device, num_classes: int):
    """原图 -> 板面前景 mask (bool 数组, 原图尺寸; True=板面)

    预处理与 train_seg.py 的 SegDataset 一致: BILINEAR resize + ImageNet 归一化;
    输出用 NEAREST 还原回原图尺寸。
    """
    w, h = img.size
    x = img.resize((imgsz, imgsz), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(x, dtype=np.float32) / 255.0).permute(2, 0, 1)
    t = (t - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    logits = model(t.unsqueeze(0).to(device))
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    pred = np.asarray(Image.fromarray(pred).resize((w, h), Image.NEAREST))
    return pred > 0  # 非背景 = 板面 (二值任务)


def box_panel_ratio(fg: np.ndarray, box) -> float:
    """框内板面像素占比 (0~1)"""
    h, w = fg.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(fg[y1:y2, x1:x2].mean())


def load_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # 旧版 Pillow 不支持 size 参数
        return ImageFont.load_default()


def render_result(img: Image.Image, fg: np.ndarray, dets: list, out_path: Path):
    """可视化: 原图 + 板面半透明叠加 + 检测框 (红=板上, 灰=疑似背景误检)"""
    base = np.asarray(img, dtype=np.float32).copy()
    sel = fg[..., None]
    base = np.where(sel, base * 0.65 + np.array(PANEL_COLOR, dtype=np.float32) * 0.35, base)
    canvas = Image.fromarray(base.astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    line_w = max(2, canvas.width // 400)
    font = load_font(max(12, canvas.width // 50))

    for d in dets:
        color = RED if d["on_panel"] else GRAY
        x1, y1, x2, y2 = (int(round(v)) for v in d["xyxy"])
        for i in range(line_w):  # 加粗方框
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
        label = f"{d['name']} {d['conf']:.2f} r={d['ratio']:.2f}"
        if not d["on_panel"]:
            label += " [bg?]"
        ty = max(0, y1 - (max(12, canvas.width // 50) + 4))
        draw.rectangle(draw.textbbox((x1, ty), label, font=font), fill=color)
        draw.text((x1, ty), label, fill=(255, 255, 255), font=font)

    legend = "green=panel  red=defect  gray=suspect(bg)"
    draw.rectangle(draw.textbbox((6, 6), legend, font=font), fill=(0, 0, 0))
    draw.text((6, 6), legend, fill=(255, 255, 255), font=font)
    canvas.save(out_path)


def main():
    args = parse_args()
    det_weights = Path(args.det_weights).expanduser().resolve()
    seg_weights = Path(args.seg_weights).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    if not det_weights.exists():
        raise SystemExit(f"错误: 找不到检测权重 {det_weights}")
    images = collect_images(Path(args.source).expanduser().resolve(), args.limit)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    print(f"PyTorch {torch.__version__} | device={device}")

    seg_model, classes, arch, num_classes, imgsz_seg = load_seg_model(seg_weights, device, args.imgsz_seg)
    print(f"分割: HRNet-{arch.upper()} | 类别 {classes} | imgsz={imgsz_seg}")
    det_model = YOLO(str(det_weights))
    print(f"检测: {det_weights.name} | 类别 {det_model.names} | imgsz={args.imgsz_det}")
    print(f"待处理: {len(images)} 张 -> {out_dir}\n")

    total, on_panel_cnt, dropped = 0, 0, 0
    records = []
    for i, img_path in enumerate(images, 1):
        img = Image.open(img_path).convert("RGB")
        fg = segment_panel(seg_model, img, imgsz_seg, device, num_classes)

        result = det_model.predict(
            str(img_path), imgsz=args.imgsz_det, conf=args.conf, iou=args.iou,
            device=device, verbose=False,
        )[0]

        dets = []
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            name = result.names.get(int(box.cls[0]), str(int(box.cls[0])))
            ratio = box_panel_ratio(fg, xyxy)
            on_panel = ratio >= args.min_panel_ratio
            total += 1
            on_panel_cnt += int(on_panel)
            if on_panel or not args.drop_background:
                dets.append({"xyxy": xyxy, "conf": conf, "name": name, "ratio": ratio,
                             "on_panel": on_panel})
            else:
                dropped += 1
            records.append([img_path.name, name, f"{conf:.4f}",
                            *[f"{v:.1f}" for v in xyxy], f"{ratio:.4f}",
                            "on-panel" if on_panel else "suspect-bg"])

        render_result(img, fg, dets, out_dir / f"{img_path.stem}_cascade.png")
        panel_pct = fg.mean() * 100
        print(f"[{i}/{len(images)}] {img_path.name} | 板面 {panel_pct:.1f}% | "
              f"框 {len(dets)}(板上 {sum(d['on_panel'] for d in dets)})")

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "conf", "x1", "y1", "x2", "y2", "panel_ratio", "verdict"])
        w.writerows(records)

    print(f"\n汇总: {len(images)} 张 | 总框 {total} | 板上 {on_panel_cnt} | "
          f"疑似背景误检 {total - on_panel_cnt}" + (f" (已丢弃 {dropped})" if args.drop_background else ""))
    print(f"可视化: {out_dir} | 明细: {csv_path}")


if __name__ == "__main__":
    main()
