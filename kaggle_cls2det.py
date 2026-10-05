"""
kaggle_cls2det.py — 把「分类文件夹」数据集转换为 YOLO 检测格式

适用场景:
    数据来源如 Kaggle: pythonafroz/solar-panel-images（光伏板六类状态图片，已按文件夹分好类）
    目录结构假定为（实际文件夹名以你下载到的为准）:
        solar-panel-images/
        ├── Clean/
        ├── Dusty/
        ├── Bird-drop/
        ├── Electrical-damage/
        ├── Physical-Damage/
        └── Snow-Covered/

转换规则:
    1. 每个子文件夹视为一个类别, 类别编号 = 文件夹名排序后的序号
    2. 每张图片生成一个「整图大框」标注 (cls 0.5 0.5 1.0 1.0)
       —— 让检测模型先学会识别「整块板的状态」, 属于快速跑通全流程的基线方案;
       若后续需要精确定位(如只框出板面灰尘区域), 需在 Roboflow / Label Studio 手工标注真实边框
    3. 自动按比例划分 train / val / test, 并生成 data.yaml

用法:
    python kaggle_cls2det.py --src /path/to/solar-panel-images --dst ./datasets/solar_cls2det
"""

import argparse
import json
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="分类文件夹数据集 → YOLO 检测格式")
    p.add_argument("--src", required=True, help="分类数据集根目录（内含各分类别文件夹）")
    p.add_argument("--dst", default="./datasets/solar_cls2det", help="输出目录")
    p.add_argument("--val", type=float, default=0.15, help="验证集比例")
    p.add_argument("--test", type=float, default=0.15, help="测试集比例")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


def copy_pair(img_file: Path, cls_id: int, split: str, dst: Path) -> None:
    """复制图片到目标目录, 并写入整图框标签（处理重名冲突）"""
    img_dir = dst / "images" / split
    lab_dir = dst / "labels" / split
    target = img_dir / img_file.name
    i = 1
    while target.exists():
        target = img_dir / f"{img_file.stem}_{i}{img_file.suffix}"
        i += 1
    shutil.copy2(img_file, target)
    (lab_dir / f"{target.stem}.txt").write_text(f"{cls_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")


def main():
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"错误: 找不到目录 {src}")

    rng = random.Random(args.seed)
    class_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"错误: {src} 下没有子文件夹, 请检查 --src 是否指向类别文件夹的上一级")
    if len(class_dirs) == 1:
        print(f"警告: 只发现 1 个类别文件夹 [{class_dirs[0].name}]。"
              "若数据集本应有多类, 请检查 --src 是否指错层级（常见于 zip 解压多了一层目录）")

    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"发现 {len(class_dirs)} 个类别: {[d.name for d in class_dirs]}")
    for cls_id, cls_dir in enumerate(class_dirs):
        files = sorted(p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not files:
            print(f"  [{cls_dir.name}] 0 张图片, 跳过")
            continue
        rng.shuffle(files)
        n = len(files)
        n_test, n_val = int(n * args.test), int(n * args.val)
        splits = {
            "test": files[:n_test],
            "val": files[n_test:n_test + n_val],
            "train": files[n_test + n_val:],
        }
        for split, split_files in splits.items():
            for f in split_files:
                copy_pair(f, cls_id, split, dst)
            print(f"  [{cls_dir.name}] {split}: {len(split_files)} 张")

    # 生成 data.yaml（path 用绝对路径, 跨环境可用）
    yaml_path = dst / "data.yaml"
    names_block = "".join(f"  {i}: {json.dumps(d.name, ensure_ascii=False)}\n"
                          for i, d in enumerate(class_dirs))
    yaml_path.write_text(
        "# 由 kaggle_cls2det.py 自动生成 (整图框近似标注)\n"
        f"path: {dst.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + names_block,
        encoding="utf-8",
    )
    print(f"\n完成! data.yaml: {yaml_path}")
    print(f"下一步: python check_dataset.py --data {yaml_path}")


if __name__ == "__main__":
    main()
