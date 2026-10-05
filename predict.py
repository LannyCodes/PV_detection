"""
predict.py — 光伏板多类检测推理

支持: 单张图片 / 图片文件夹 / 视频文件, 结果（带框可视化）自动保存
Notebook 提示: 加 --show 可在 Colab/Kaggle 中直接内联展示每张结果图

用法:
    python predict.py --weights runs/pv_detect/train/weights/best.pt --source test.jpg --show
    python predict.py --weights ... --source test_images/          # 整个文件夹
    python predict.py --weights ... --source video.mp4             # 视频（本地）
    python predict.py --weights yolo11n.pt --source img.jpg        # 用通用模型先试跑
"""

import argparse
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="光伏板多类检测推理")
    p.add_argument("--weights", default="runs/pv_detect/train/weights/best.pt", help="权重文件")
    p.add_argument("--source", required=True, help="图片 / 文件夹 / 视频 路径（本地摄像头可传 0）")
    p.add_argument("--conf", type=float, default=0.4, help="置信度阈值")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto", help="auto / 0 / cpu / mps")
    p.add_argument("--project", default="runs/pv_detect")
    p.add_argument("--name", default="predict")
    p.add_argument("--show", action="store_true", help="Notebook 中内联展示结果")
    return p.parse_args()


def pick_device(arg: str):
    import torch
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def show_inline(results):
    """在 Notebook 环境内联展示每张结果图"""
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        print("提示: 当前环境无 IPython, --show 已忽略（结果图已存磁盘）")
        return
    if get_ipython() is None:
        print("提示: 当前不是 Notebook 环境, --show 已忽略（结果图已存磁盘）")
        return
    for r in results:
        p = Path(r.save_dir) / Path(r.path).name
        if p.exists() and p.suffix.lower() in IMG_EXTS:
            display(Image(filename=str(p), width=560))


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"错误: 找不到权重 {weights}\n"
            "请先运行 train.py；或临时用 --weights yolo11n.pt 只做通用试跑"
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("缺少 ultralytics, 请先执行: pip install ultralytics")

    device = pick_device(args.device)
    model = YOLO(str(weights))
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    # 汇总统计
    counter = Counter()
    for r in results:
        if r.boxes is None:
            continue
        for cls_id in r.boxes.cls.tolist():
            counter[model.names[int(cls_id)]] += 1
    print(f"\n共处理 {len(results)} 个输入 | 检测统计: {dict(counter) if counter else '未检出目标'}")
    if results:
        print(f"结果已保存到: {Path(results[0].save_dir)}")

    if args.show:
        show_inline(results)


if __name__ == "__main__":
    main()
