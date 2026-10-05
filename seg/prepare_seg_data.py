"""
prepare_seg_data.py — 分割数据集准备: 公开数据集 → 统一语义分割训练格式

把 Roboflow 等公开数据集转换为分割训练直接可用的格式:

    dst/
    ├── images/{train,val}/*.jpg   原图
    ├── masks/{train,val}/*.png    单通道类别掩码 (像素值 = 类别 id, 0 = 背景)
    ├── meta.json                  类别表 / 划分统计 / 像素统计
    └── preview/*.png              掩码叠加可视化抽查

支持两种源格式 (自动探测):
    A. YOLO-seg 多边形标注 (Roboflow 导出 "YOLOv8/YOLO11-seg" 格式):
       dataset-root/
       ├── train/images/*.jpg + train/labels/*.txt   # 归一化多边形: cls x1 y1 x2 y2 ...
       ├── valid/images/...                          # 缺失 val 时自动从 train 切分
       └── data.yaml                                 # 可选, 用于读取类别名

    B. 图像-掩码配对 (常见于学术公开分割数据集):
       dataset-root/
       ├── {train,...}/images/*.jpg
       └── {train,...}/masks/*.png    # 掩码也可在根级 / 目录名为 labels, gt;
                                      # 也支持与图同名的 *_mask.png 命名

用法:
    # Roboflow 板面分割导出包 → 只保留 "背景/板" 两类 (所有非背景类合并)
    python seg/prepare_seg_data.py --src ~/Downloads/solar-panel-seg --dst seg/datasets/panel_seg --binary

    # 保留原始多类标注
    python seg/prepare_seg_data.py --src ~/Downloads/xxx --dst seg/datasets/xxx

说明:
    - YOLO-seg 的类别 id 统一 +1 偏移 (0 保留给背景); 已有 png 掩码不做偏移
    - --binary 时所有非背景像素统一为 1
    - 转换后务必查看 preview/ 抽查掩码质量
"""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import yaml
except ImportError:  # data.yaml 仅用于读取类别名, 缺失不影响转换
    yaml = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
IMG_DIR_NAMES = ("images", "img", "image")
MASK_DIR_NAMES = ("masks", "mask", "labels", "gt", "ground_truth")

# 预览叠加色 (按类别 id 取用, 0 = 背景不画)
PALETTE = [
    (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0), (175, 82, 222),
    (255, 214, 10), (90, 200, 250), (255, 105, 180), (48, 209, 88), (94, 92, 230),
    (255, 179, 64), (172, 142, 104), (142, 142, 147), (99, 230, 226), (250, 17, 79),
    (162, 132, 94), (106, 76, 147), (0, 199, 190), (255, 45, 85), (88, 86, 214),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="公开分割数据集 → 统一训练格式 (images + masks PNG + meta.json)")
    p.add_argument("--src", required=True, help="源数据集目录 (Roboflow 导出根目录, 或 images/masks 配对目录)")
    p.add_argument("--dst", default="seg/datasets/seg_dataset", help="输出目录")
    p.add_argument("--val", type=float, default=0.15, help="无现成划分时, 从训练数据切出的验证集比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--binary", action="store_true", help="所有非背景类合并为 1 类 (如 板/背景)")
    p.add_argument("--preview", type=int, default=8, help="掩码叠加可视化抽查张数 (0 关闭)")
    p.add_argument("--force", action="store_true", help="输出目录非空时清空重建")
    return p.parse_args()


def color_of(cid: int):
    if 1 <= cid <= len(PALETTE):
        return PALETTE[cid - 1]
    return ((cid * 53) % 156 + 60, (cid * 97) % 156 + 60, (cid * 193) % 156 + 60)


def list_images(directory: Path, exclude: Path = None):
    """递归列出图像文件 (可排除某个子树, 避免扫进掩码目录)"""
    out = []
    for p in sorted(directory.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if exclude is not None and exclude in p.parents:
            continue
        out.append(p)
    return out


def detect_yolo_seg(src: Path) -> dict:
    """探测 YOLO-seg 多边形标注结构 → {split: (img_dir, label_dir)}"""
    found = {}
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        key = SPLIT_ALIASES.get(d.name.lower())
        if key is None:
            continue
        label_dir = d / "labels"
        if label_dir.is_dir() and any(label_dir.glob("*.txt")):
            img_dir = d / "images" if (d / "images").is_dir() else d
            found[key] = (img_dir, label_dir)
    if not found:
        # 根级 images + labels 无划分的情形
        label_dir, img_dir = src / "labels", src / "images"
        if img_dir.is_dir() and label_dir.is_dir() and any(label_dir.glob("*.txt")):
            found["train"] = (img_dir, label_dir)
    return found


def detect_pair(src: Path) -> dict:
    """探测图像-掩码配对结构 → {split: (img_dir, mask_dir)}"""
    found = {}
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        key = SPLIT_ALIASES.get(d.name.lower())
        if key is None:
            continue
        img_dir = next((d / c for c in IMG_DIR_NAMES if (d / c).is_dir()), None)
        mask_dir = next((d / c for c in MASK_DIR_NAMES if (d / c).is_dir()), None)
        if img_dir is None and mask_dir is not None:
            img_dir = d  # 图像直接放在 split 目录下
        if img_dir and mask_dir:
            found[key] = (img_dir, mask_dir)
    if found:
        return found
    # 根级 images + masks
    img_dir = next((src / c for c in IMG_DIR_NAMES if (src / c).is_dir()), None)
    mask_dir = next((src / c for c in MASK_DIR_NAMES if (src / c).is_dir()), None)
    if img_dir is None and mask_dir is not None:
        img_dir = src
    if img_dir and mask_dir:
        return {"train": (img_dir, mask_dir)}
    return {}


def collect_yolo(found: dict) -> dict:
    """{split: (img_dir, label_dir)} → {split: [(img, label_txt, 'poly')]}"""
    entries = {}
    for split, (img_dir, label_dir) in found.items():
        entries[split] = [
            (img, label_dir / f"{img.stem}.txt", "poly")
            for img in list_images(img_dir, exclude=label_dir)
        ]
    return entries


def collect_pairs(found: dict):
    """{split: (img_dir, mask_dir)} → ({split: [(img, mask, 'mask')]}, 未配对数)"""
    entries, unmatched = {}, 0
    for split, (img_dir, mask_dir) in found.items():
        # 掩码索引: stem → 路径 (兼容 *_mask / *_gt 命名)
        mask_index = {}
        for m in mask_dir.rglob("*"):
            if not m.is_file():
                continue
            mask_index.setdefault(m.stem, m)
            stem_low = m.stem.lower()
            for suffix in ("_mask", "_gt", "_label"):
                if stem_low.endswith(suffix):
                    mask_index.setdefault(m.stem[: -len(suffix)], m)
        items = []
        for img in list_images(img_dir, exclude=mask_dir):
            mask = mask_index.get(img.stem)
            if mask is None:
                unmatched += 1
                continue
            items.append((img, mask, "mask"))
        entries[split] = items
    return entries, unmatched


def plan_splits(entries: dict, val_ratio: float, seed: int) -> dict:
    """已有 train+val 直接沿用; 只有 train 则按比例切出 val"""
    if not entries.get("train"):
        raise SystemExit("错误: 未找到训练数据 (train 划分)")
    if entries.get("val"):
        return {"train": entries["train"], "val": entries["val"]}
    train_all = entries["train"]
    if len(train_all) < 2:
        raise SystemExit("错误: 训练数据过少, 无法切出验证集")
    rng = random.Random(seed)
    idx = list(range(len(train_all)))
    rng.shuffle(idx)
    n_val = min(len(train_all) - 1, max(1, int(len(train_all) * val_ratio)))
    val_idx = set(idx[:n_val])
    return {
        "train": [item for i, item in enumerate(train_all) if i not in val_idx],
        "val": [train_all[i] for i in sorted(val_idx)],
    }


def rasterize_yolo_label(label_path: Path, width: int, height: int, binary: bool):
    """YOLO-seg 多边形 txt → 类别掩码 (uint8)

    返回 (mask, 出现类别集合, 坏行数)。类别 id 统一 +1 偏移 (0 留给背景);
    binary 模式下所有标注统一为 1。标签文件缺失时返回全背景掩码。
    """
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    classes, bad = set(), 0
    if not label_path.exists():
        return np.array(mask_img), classes, bad
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            bad += 1
            continue
        if len(coords) < 6 or len(coords) % 2 != 0:
            bad += 1
            continue
        cid = 1 if binary else cls + 1
        if cid > 255:
            bad += 1
            continue
        pts = [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)]
        draw.polygon(pts, fill=cid)
        classes.add(cid)
    return np.array(mask_img), classes, bad


def load_orig_mask(mask_path: Path, width: int, height: int, binary: bool) -> np.ndarray:
    """读取已有 png 掩码 → uint8 数组 (不做 id 偏移, 尊重原标注); 尺寸不符时最近邻缩放"""
    m = Image.open(mask_path)
    if m.mode not in ("L", "P", "1"):
        print(f"  警告: 掩码 {mask_path.name} 为 {m.mode} 模式, 按灰度读取 (类别 id 可能失真)")
        m = m.convert("L")
    if m.size != (width, height):
        m = m.resize((width, height), Image.NEAREST)
    arr = np.array(m)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if binary:
        arr = (arr > 0).astype(np.uint8)
    return arr


def read_names(src: Path):
    """从 data.yaml 读取类别名 (Roboflow 导出通常带), 读不到返回 None"""
    yaml_path = src / "data.yaml"
    if yaml is None or not yaml_path.exists():
        return None
    try:
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return None
    names = cfg.get("names") if isinstance(cfg, dict) else None
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):
        return names
    return None


def save_preview(img_path: Path, mask: np.ndarray, out_path: Path, alpha: float = 0.45):
    """掩码半透明叠加到原图, 供人工抽查"""
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
    for cid in np.unique(mask):
        if cid == 0:
            continue
        sel = mask == cid
        color = np.array(color_of(int(cid)), dtype=np.float32)
        img[sel] = img[sel] * (1 - alpha) + color * alpha
    Image.fromarray(img.astype(np.uint8)).save(out_path)


def main():
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"错误: 找不到源目录 {src}")

    # ---- 1. 探测格式并收集条目 ----
    yolo_found = detect_yolo_seg(src)
    if yolo_found:
        fmt = "yolo-seg"
        entries = collect_yolo(yolo_found)
        unmatched = 0
    else:
        pair_found = detect_pair(src)
        if not pair_found:
            raise SystemExit(
                f"错误: 无法识别数据集结构: {src}\n"
                "期望结构之一:\n"
                "  A. YOLO-seg: train(/valid/test)/images + 同级 labels/*.txt (+ data.yaml)\n"
                "  B. 图像-掩码配对: images/ + masks(png)  [可在根级或 train/valid 子目录内]"
            )
        fmt = "pair"
        entries, unmatched = collect_pairs(pair_found)

    if "test" in entries:
        n_test = len(entries.pop("test"))
        print(f"提示: 忽略 test 划分 ({n_test} 张), 本项目训练只用 train/val")

    # ---- 2. 类别名 (可选) ----
    names = read_names(src)
    if names:
        print(f"data.yaml 类别: {names}")

    # ---- 3. 划分 ----
    plan = plan_splits(entries, args.val, args.seed)

    # ---- 4. 输出目录准备 ----
    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            raise SystemExit(f"错误: 输出目录非空: {dst}\n如需覆盖请加 --force")
        for name in ("images", "masks", "preview", "meta.json"):
            p = dst / name
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()
    for split in plan:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "masks" / split).mkdir(parents=True, exist_ok=True)

    # ---- 5. 转换主循环 ----
    stats = {
        split: {"empty_masks": 0, "total_pixels": 0, "class_images": Counter(), "class_pixels": Counter()}
        for split in plan
    }
    bad_lines = missing_labels = 0
    saved = {split: [] for split in plan}  # (img_out, mask_out), 供预览抽样

    done, total = 0, sum(len(items) for items in plan.values())
    for split, items in plan.items():
        img_out_dir = dst / "images" / split
        mask_out_dir = dst / "masks" / split
        st = stats[split]
        for img_path, payload, kind in items:
            done += 1
            if done % 500 == 0:
                print(f"  ...处理中 {done}/{total}")
            with Image.open(img_path) as im:
                width, height = im.size

            if kind == "poly":
                mask, _, bad = rasterize_yolo_label(payload, width, height, args.binary)
                bad_lines += bad
                if not payload.exists():
                    missing_labels += 1
            else:
                mask = load_orig_mask(payload, width, height, args.binary)

            # 输出文件名 (重名时加序号, 图像与掩码保持同名)
            stem = img_path.stem
            i = 1
            while (img_out_dir / f"{stem}{img_path.suffix.lower()}").exists():
                stem = f"{img_path.stem}_{i}"
                i += 1
            target_img = img_out_dir / f"{stem}{img_path.suffix.lower()}"
            target_mask = mask_out_dir / f"{stem}.png"

            shutil.copy2(img_path, target_img)
            Image.fromarray(mask).save(target_mask)

            if mask.max() == 0:
                st["empty_masks"] += 1
            st["total_pixels"] += width * height
            for cid in np.unique(mask):
                if cid == 0:
                    continue
                st["class_images"][int(cid)] += 1
                st["class_pixels"][int(cid)] += int((mask == cid).sum())
            saved[split].append((target_img, target_mask))

    # ---- 6. 类别表与 meta.json ----
    if args.binary:
        classes_out = {"0": "background", "1": "foreground"}
    elif fmt == "yolo-seg" and names:
        classes_out = {"0": "background"}
        classes_out.update({str(int(cid) + 1): str(nm) for cid, nm in names.items()})
    else:
        classes_out = {"0": "background"}
        ids = set()
        for st in stats.values():
            ids |= set(st["class_pixels"])
        for cid in sorted(ids):
            classes_out[str(cid)] = f"class_{cid}"

    meta = {
        "task": "semantic_segmentation",
        "source": str(src),
        "source_format": fmt,
        "binary": args.binary,
        "classes": classes_out,
        "splits": {split: len(items) for split, items in plan.items()},
        "stats": {
            split: {
                "empty_masks": st["empty_masks"],
                "total_pixels": st["total_pixels"],
                "class_images": {str(k): int(v) for k, v in sorted(st["class_images"].items())},
                "class_pixels": {str(k): int(v) for k, v in sorted(st["class_pixels"].items())},
            }
            for split, st in stats.items()
        },
        "created_by": "seg/prepare_seg_data.py",
    }
    meta_path = dst / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 7. 预览抽查 ----
    n_preview = 0
    if args.preview > 0:
        preview_dir = dst / "preview"
        preview_dir.mkdir(exist_ok=True)
        pool = list(saved.get("val") or [])
        if len(pool) < args.preview:
            pool += saved.get("train") or []
        step = max(1, len(pool) // args.preview)
        for img_out, mask_out in pool[::step][: args.preview]:
            mask = np.array(Image.open(mask_out))
            # 文件名带 split 前缀, 避免 train/val 同名图互相覆盖
            save_preview(img_out, mask, preview_dir / f"check_{img_out.parent.name}_{img_out.stem}.png")
            n_preview += 1

    # ---- 8. 汇总 ----
    print("\n===== 准备完成 =====")
    print(f"源格式: {fmt} | 输出: {dst}")
    for split, items in plan.items():
        st = stats[split]
        ratio = ", ".join(
            f"{cid}:{px / max(1, st['total_pixels']) * 100:.1f}%"
            for cid, px in sorted(st["class_pixels"].items())
        )
        print(f"[{split}] {len(items)} 张 | 空掩码(全背景): {st['empty_masks']} | 类别像素占比: {ratio or '-'}")
    if bad_lines:
        print(f"警告: 跳过坏标注行 {bad_lines} 条")
    if missing_labels:
        print(f"警告: {missing_labels} 张图像无对应标签文件 (已按全背景处理)")
    if unmatched:
        print(f"警告: {unmatched} 张图像未找到配对掩码 (已跳过)")
    if n_preview:
        print(f"抽查预览: {dst / 'preview'} ({n_preview} 张, 建议人工核对掩码质量)")
    print(f"meta: {meta_path}")
    print(f"\n下一步: python seg/train_seg.py --data {args.dst}")


if __name__ == "__main__":
    main()
