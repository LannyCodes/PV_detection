"""
check_dataset.py — YOLO 数据集体检工具（训练前先跑, 避免浪费 GPU 时间）

功能:
    - 读取 data.yaml, 统计 train/val/test 各划分的图像数与标注数
    - 统计每个类别的框数量, 快速发现类别不平衡
    - 检查: 有图无标签 / 空标签 / 类别编号越界 / 标注行格式错误

data.yaml 格式（Roboflow 导出或 kaggle_cls2det.py 生成）:
    path: /数据集根目录
    train: images/train
    val: images/val
    test: images/test
    names:
      0: Clean
      1: Dust
      ...

用法:
    python check_dataset.py --data ./datasets/solar_cls2det/data.yaml
"""

import argparse
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="YOLO 数据集结构与标注体检")
    p.add_argument("--data", required=True, help="data.yaml 路径")
    return p.parse_args()


def resolve_root(yaml_path: Path, cfg: dict) -> Path:
    """确定数据集根目录: 优先用 yaml 里的 path, 否则用 yaml 所在目录"""
    raw = cfg.get("path")
    if not raw:
        return yaml_path.parent
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [yaml_path.parent / p, Path.cwd() / p]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"错误: data.yaml 中的 path 不存在: {raw}")


def check_split(root: Path, split_name: str, img_spec, names: dict) -> None:
    """检查单个划分（train/val/test）"""
    img_specs = img_spec if isinstance(img_spec, list) else [img_spec]
    n_images = n_with_label = 0
    cls_counter = Counter()
    missing, empty, bad_lines, bad_cls = [], [], 0, 0

    for spec in img_specs:
        img_dir = Path(spec)
        if not img_dir.is_absolute():
            img_dir = root / img_dir
        # 兼容 Roboflow 导出的 ../train/images 相对写法（数据实际位于数据集根目录下）
        if not img_dir.exists() and str(spec).startswith("../"):
            img_dir = root / str(spec)[3:]
        # 标签目录: 按 YOLO 标准约定, 将路径中的 images 段替换为 labels
        lab_dir = Path(str(img_dir).replace("images", "labels", 1))
        if not lab_dir.exists():
            print(f"  警告: 标签目录不存在: {lab_dir}（若结构非标准请检查）")

        imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        for img in imgs:
            n_images += 1
            lab = lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                missing.append(str(img))
                continue
            lines = [ln.strip() for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                empty.append(str(img))
                continue
            n_with_label += 1
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    bad_lines += 1
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    [float(x) for x in parts[1:]]
                except ValueError:
                    bad_lines += 1
                    continue
                if cls_id not in names:
                    bad_cls += 1
                else:
                    cls_counter[cls_id] += 1

    print(f"\n===== [{split_name}] =====")
    print(f"图像: {n_images}  |  含标注: {n_with_label}  |  "
          f"无标签文件: {len(missing)}（会被当作背景图训练)  |  空标签: {len(empty)}")
    if bad_lines or bad_cls:
        print(f"⚠ 可疑标注行: {bad_lines}  |  类别编号越界: {bad_cls}")
    if cls_counter:
        print("各类别框数:")
        for cid, cnt in sorted(cls_counter.items()):
            print(f"  {cid:>2} {str(names.get(cid, '?')):<24} {cnt}")
    for m in missing[:5]:
        print(f"  [无标签] {m}")
    if len(missing) > 5:
        print(f"  ... 其余 {len(missing) - 5} 个省略")


def main():
    args = parse_args()
    yaml_path = Path(args.data)
    if not yaml_path.exists():
        raise SystemExit(f"错误: 找不到 {yaml_path}")

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = resolve_root(yaml_path, cfg)

    names = cfg.get("names", {})
    if isinstance(names, list):  # 兼容 names 为列表的写法
        names = dict(enumerate(names))
    print(f"数据集根目录: {root}")
    print(f"类别 ({len(names)} 个): {names}")

    found_split = False
    for split in ("train", "val", "test"):
        if split in cfg:
            check_split(root, split, cfg[split], names)
            found_split = True
    if not found_split:
        raise SystemExit("错误: data.yaml 中未找到 train/val/test 字段")


if __name__ == "__main__":
    main()
