"""
train.py — 光伏板多类检测训练 (Ultralytics YOLO11)

支持环境: Google Colab / Kaggle / 本地 (CUDA / Apple MPS / CPU 自动选择)
前置:    pip install ultralytics；数据集含 data.yaml (Roboflow 导出 或 kaggle_cls2det.py 生成)

用法示例:
    python train.py --data datasets/solar_cls2det/data.yaml
    python train.py --data /content/solar-fault/data.yaml --model yolo11s.pt --epochs 150
    python train.py --data ... --resume          # 从上次中断处续训
    python train.py --data ... --export-onnx     # 训练完成后导出 ONNX（部署用）
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="YOLO11 光伏板多类检测训练")
    p.add_argument("--data", required=True, help="data.yaml 路径")
    p.add_argument("--model", default="yolo11n.pt", help="预训练权重, 可选 yolo11n/s/m/l")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=None, help="批大小; 默认 GPU=-1(自动) 其他=16")
    p.add_argument("--device", default="auto", help="auto / 0 / cpu / mps")
    p.add_argument("--project", default="runs/pv_detect", help="结果保存根目录")
    p.add_argument("--name", default="train", help="本次实验名")
    p.add_argument("--patience", type=int, default=30, help="早停耐心值")
    p.add_argument("--resume", action="store_true", help="从 project/name 的 last.pt 续训")
    p.add_argument("--export-onnx", action="store_true", help="训练完成后导出 ONNX")
    return p.parse_args()


def pick_device(arg: str):
    """自动选择推理/训练设备"""
    import torch
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(
            f"错误: 找不到 {data}\n"
            "提示: 用 kaggle_cls2det.py 转换 Kaggle 分类数据集, "
            "或指向 Roboflow 导出目录里的 data.yaml"
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("缺少 ultralytics, 请先执行: pip install ultralytics")

    import torch

    device = pick_device(args.device)
    batch = args.batch if args.batch is not None else (-1 if device == 0 else 16)
    print(f"PyTorch {torch.__version__} | device={device} | batch={batch}")

    if args.resume:
        last = Path(args.project) / args.name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"错误: 未找到可续训权重 {last}, 请先完成一次训练")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            seed=0,
        )

    save_dir = Path(getattr(model.trainer, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    print(f"\n训练结束, 结果目录: {save_dir}")
    if not best.exists():
        return
    print(f"最佳权重: {best}")

    # 用最佳权重在验证集上评估
    best_model = YOLO(str(best))
    metrics = best_model.val(data=str(data), device=device)
    if getattr(metrics, "box", None) is not None:
        print(f"验证集: mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    # 导出 ONNX（后续部署到服务端/端侧用）
    if args.export_onnx:
        onnx_path = best_model.export(format="onnx", imgsz=args.imgsz)
        print(f"ONNX 已导出: {onnx_path}")

    # Colab 持久化提示（/content 会话结束即清空）
    if Path("/content").exists() and not str(save_dir.resolve()).startswith("/content/drive"):
        print("\n提示(Colab): /content 会话结束即清空, 建议备份最佳权重到 Google Drive:")
        print(f"  !mkdir -p /content/drive/MyDrive/yolo_results && "
              f"cp {best} /content/drive/MyDrive/yolo_results/")


if __name__ == "__main__":
    main()
