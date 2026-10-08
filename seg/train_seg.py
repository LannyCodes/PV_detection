"""
train_seg.py — HRNet 语义分割训练 (板面分割 / 缺陷分割通用)

前置: 先用 prepare_seg_data.py 生成数据集 (images/masks/{train,val} + meta.json)
特性: CE+Dice 组合损失 / AdamW + warmup + cosine / mIoU 评估 / 早停 /
      AMP 混合精度 (CUDA) / 断点续训 / 训练结束自动保存预测可视化

用法示例:
    python seg/train_seg.py --data seg/datasets/panel_seg
    python seg/train_seg.py --data seg/datasets/panel_seg --arch w32 --imgsz 512 --epochs 100
    python seg/train_seg.py --data seg/datasets/panel_seg --resume        # 断点续训
    torchrun --standalone --nproc_per_node=2 seg/train_seg.py --data seg/datasets/panel_seg --batch 6   # 双卡 DDP (--batch 为每卡 batch)

输出 (project/name/, 默认 runs/pv_seg/train/):
    best.pt         验证 mIoU 最优权重 (含类别表, 供级联推理加载)
    last.pt         最近一轮权重 (供 --resume)
    results.csv     每轮指标 (epoch, lr, loss, mIoU, acc)
    predictions/    验证集抽样三联图 (原图 | GT | 预测)
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 同目录模型定义 (先加路径, 支持从任意工作目录运行)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrnet import build_hrnet  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 可视化叠加色 (与 prepare_seg_data.py 保持一致)
PALETTE = [
    (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0), (175, 82, 222),
    (255, 214, 10), (90, 200, 250), (255, 105, 180), (48, 209, 88), (94, 92, 230),
]


def parse_args():
    p = argparse.ArgumentParser(description="HRNet 语义分割训练")
    p.add_argument("--data", required=True, help="prepare_seg_data.py 生成的目录 (含 images/masks/meta.json)")
    p.add_argument("--arch", default="w18", help="HRNet 规格: w18 / w32 / w48")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=3, help="学习率预热轮数")
    p.add_argument("--workers", type=int, default=4, help="DataLoader 进程数 (Kaggle 4 核建议 2-4)")
    p.add_argument("--device", default="auto", help="auto / 0 / cpu / mps")
    p.add_argument("--project", default="runs/pv_seg")
    p.add_argument("--name", default="train")
    p.add_argument("--patience", type=int, default=20, help="早停耐心值 (0 关闭)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="从 project/name/last.pt 续训")
    return p.parse_args()


def pick_device(arg: str):
    """自动选择训练设备 (与 train.py 同一策略)"""
    if arg != "auto":
        return int(arg) if str(arg).isdigit() else arg
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class SegDataset(Dataset):
    """读取 prepare_seg_data.py 的输出: images/{split}/*.jpg + masks/{split}/*.png"""

    def __init__(self, root: Path, split: str, imgsz: int, augment: bool = False):
        self.img_dir = root / "images" / split
        self.mask_dir = root / "masks" / split
        if not self.img_dir.is_dir():
            raise SystemExit(f"错误: 找不到 {self.img_dir}, 请先运行 seg/prepare_seg_data.py")
        self.images = sorted(
            p for p in self.img_dir.iterdir() if p.suffix.lower() in IMG_EXTS
        )
        if not self.images:
            raise SystemExit(f"错误: {self.img_dir} 下没有图像")
        self.imgsz = imgsz
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            raise SystemExit(f"错误: 图像 {img_path.name} 缺少对应掩码 {mask_path.name}")
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.augment:
            # 仅几何增强; 不做色彩抖动 (颜色对灰尘/积雪判别有语义)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            k = random.choice((0, 0, 1, 2, 3))  # 1/4 概率不旋转
            if k:
                img = img.rotate(90 * k, expand=True)
                mask = mask.rotate(90 * k, expand=True)

        img = img.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        mask = mask.resize((self.imgsz, self.imgsz), Image.NEAREST)  # 类别 id 不能插值

        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        img_t = (img_t - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        return img_t, mask_t


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6):
    """多类 soft Dice (对所有类别平均)"""
    probs = logits.softmax(dim=1)
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    union = probs.sum(dims) + onehot.sum(dims)
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


class ComboLoss(nn.Module):
    """交叉熵 + Dice 各半"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, logits, target):
        return 0.5 * self.ce(logits, target) + 0.5 * dice_loss(logits, target, self.num_classes)


def set_learning_rate(optimizer, base_lr, epoch, epochs, warmup):
    """warmup + cosine 衰减 (按 epoch 直接计算, 续训天然可恢复)"""
    if warmup > 0 and epoch < warmup:
        factor = (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(1, epochs - warmup)
        progress = min(max(progress, 0.0), 1.0)
        factor = 0.5 * (1 + math.cos(math.pi * progress))
    lr = base_lr * factor
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp, scaler):
    model.train()
    loss_sum, n_batches = 0.0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss = criterion(model(imgs), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item()
        n_batches += 1
    return loss_sum / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, num_classes, device, use_amp):
    """混淆矩阵统计: 返回 (mIoU, pixel_acc, {类别: IoU}, val_loss)"""
    model.eval()
    criterion = ComboLoss(num_classes)
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)  # (GT, 预测)
    loss_sum, n_batches = 0.0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, masks)
        loss_sum += loss.item()
        n_batches += 1
        pred = logits.argmax(1).cpu()
        gt = masks.cpu()
        idx = gt.flatten() * num_classes + pred.flatten()
        conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    diag = conf.diag().float()
    row = conf.sum(1).float()  # 各类 GT 像素数
    col = conf.sum(0).float()  # 各类预测像素数
    iou = diag / (row + col - diag).clamp(min=1)
    present = row > 0
    miou = iou[present].mean().item() if present.any() else 0.0
    pix_acc = (diag.sum() / conf.sum().clamp(min=1)).item()
    per_class = {int(c): iou[c].item() for c in range(num_classes) if present[c]}
    return miou, pix_acc, per_class, loss_sum / max(1, n_batches)


def build_palette(classes: dict) -> dict:
    palette = {}
    for cid in sorted(int(k) for k in classes):
        palette[cid] = (0, 0, 0) if cid == 0 else PALETTE[(cid - 1) % len(PALETTE)]
    return palette


def overlay_mask(img: np.ndarray, mask: np.ndarray, palette: dict, alpha: float = 0.5):
    out = img.astype(np.float32).copy()
    for cid, color in palette.items():
        if cid == 0:
            continue
        sel = mask == cid
        if sel.any():
            out[sel] = out[sel] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return out.astype(np.uint8)


@torch.no_grad()
def save_predictions(model, dataset, device, out_dir: Path, palette: dict, n: int = 4):
    """验证集均匀抽样: 原图 | GT | 预测 三联图, 供人工检查分割效果"""
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(dataset) // n)
    picks = list(range(0, len(dataset), step))[:n]
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    for i in picks:
        img_t, mask_t = dataset[i]
        pred = model(img_t.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
        gt = mask_t.numpy().astype(np.uint8)
        img = (img_t.numpy() * std + mean).clip(0, 1)
        img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
        sep = np.full((img.shape[0], 4, 3), 255, dtype=np.uint8)
        combo = np.concatenate(
            [img, sep, overlay_mask(img, gt, palette), sep, overlay_mask(img, pred, palette)], axis=1
        )
        Image.fromarray(combo).save(out_dir / f"pred_{i:04d}.png")
    print(f"预测可视化已保存: {out_dir} (左=原图, 中=GT, 右=预测)")


def main():
    args = parse_args()
    data_root = Path(args.data).expanduser().resolve()
    meta_path = data_root / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"错误: 找不到 {meta_path}, 请先运行 seg/prepare_seg_data.py 生成数据集")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classes = meta.get("classes", {"0": "background"})
    num_classes = max(int(k) for k in classes) + 1

    # 多卡: 通过 torchrun --nproc_per_node=N 启动时自动启用 DDP (LOCAL_RANK 由 torchrun 注入)
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        is_main = dist.get_rank() == 0
        device = local_rank
    else:
        world_size, is_main, device = 1, True, pick_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_amp = isinstance(device, int)  # CUDA (含 DDP 各卡) 启用 AMP
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if is_main:
        print(f"数据集: {data_root} | 类别 {num_classes} 个: {classes}")
        print(f"PyTorch {torch.__version__} | device={device} | world_size={world_size} | "
              f"AMP={'on' if use_amp else 'off'}")

    train_set = SegDataset(data_root, "train", args.imgsz, augment=True)
    val_set = SegDataset(data_root, "val", args.imgsz, augment=False)
    train_sampler = None
    if ddp:
        train_sampler = torch.utils.data.DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoader(
            train_set, batch_size=args.batch, sampler=train_sampler,
            num_workers=args.workers, pin_memory=True,
            drop_last=len(train_set) >= args.batch * world_size,
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=args.batch, shuffle=True,
            num_workers=args.workers, pin_memory=(device == 0),
            drop_last=len(train_set) >= args.batch,
        )
    # 验证只在主进程执行 (val_loader 仅主进程构建)
    val_loader = None
    if is_main:
        val_loader = DataLoader(
            val_set, batch_size=max(1, args.batch // 2), shuffle=False,
            num_workers=args.workers, pin_memory=(device == 0),
        )
        print(f"train: {len(train_set)} 张 | val: {len(val_set)} 张 | 每卡 batch={args.batch} | 全局 batch={args.batch * world_size}")

    model = build_hrnet(args.arch, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"模型: HRNet-{args.arch.upper()} ({n_params:.1f}M 参数)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = ComboLoss(num_classes)

    save_dir = Path(args.project) / args.name
    ckpt_last, ckpt_best = save_dir / "last.pt", save_dir / "best.pt"
    start_epoch, best_miou, patience_cnt = 0, -1.0, 0
    if args.resume:
        if not ckpt_last.exists():
            raise SystemExit(f"错误: 未找到可续训权重 {ckpt_last}, 请先完成一次训练")
        ckpt = torch.load(ckpt_last, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", -1.0)
        if is_main:
            print(f"已加载 {ckpt_last}, 从 epoch {start_epoch + 1} 继续 (best mIoU={best_miou:.4f})")

    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    if start_epoch >= args.epochs and is_main:
        print(f"提示: 已完成 {start_epoch} 轮 >= 目标轮数 {args.epochs}, 如需继续请调大 --epochs")

    log_file, writer = None, None
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "results.csv"
        log_file = open(log_path, "a" if args.resume else "w", newline="", encoding="utf-8")
        writer = csv.writer(log_file)
        if not args.resume:
            writer.writerow(["epoch", "lr", "train_loss", "val_loss", "mIoU", "pixel_acc"])

    palette = build_palette(classes)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        lr_now = set_learning_rate(optimizer, args.lr, epoch, args.epochs, args.warmup)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp, scaler)

        # 验证与日志只在主进程执行; 下一轮训练开头的 DDP allreduce 会隐式同步其余 rank
        if is_main:
            miou, pix_acc, per_class, val_loss = evaluate(model, val_loader, num_classes, device, use_amp)
            miou_t = torch.tensor([miou], device=device)
        else:
            miou_t = torch.tensor([0.0], device=device)
        if ddp:
            dist.broadcast(miou_t, src=0)  # 广播指标, 保证各 rank 的早停决策一致
        miou = miou_t.item()

        improved = miou > best_miou
        if improved:
            best_miou, patience_cnt = miou, 0
        else:
            patience_cnt += 1

        if is_main:
            writer.writerow([epoch + 1, f"{lr_now:.6f}", f"{train_loss:.4f}",
                             f"{val_loss:.4f}", f"{miou:.4f}", f"{pix_acc:.4f}"])
            log_file.flush()
            flag = " *best*" if improved else ""
            print(f"[{epoch + 1}/{args.epochs}] lr={lr_now:.5f} loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"mIoU={miou:.4f} acc={pix_acc:.4f}{flag} {time.time() - t0:.0f}s")
            if improved:
                iou_str = " ".join(f"{classes.get(str(cid), cid)}:{v:.3f}" for cid, v in sorted(per_class.items()))
                print(f"    各类 IoU: {iou_str}")

            state = {
                "model": (model.module if ddp else model).state_dict(),  # 存原始结构, 供单卡/推理加载
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
                "arch": args.arch,
                "num_classes": num_classes,
                "classes": classes,
                "imgsz": args.imgsz,
            }
            torch.save(state, ckpt_last)
            if improved:
                torch.save(state, ckpt_best)

        if args.patience > 0 and patience_cnt >= args.patience:
            if is_main:
                print(f"早停: 连续 {args.patience} 轮 mIoU 无提升")
            break

    if is_main:
        log_file.close()
        print(f"\n训练结束: 最佳 mIoU={best_miou:.4f}")
        print(f"权重: {ckpt_best} | 日志: {log_path}")
        if ckpt_best.exists():
            ckpt = torch.load(ckpt_best, map_location="cpu")
            net = model.module if ddp and hasattr(model, "module") else model
            net.load_state_dict(ckpt["model"])
            save_predictions(net, val_set, device, save_dir / "predictions", palette)
    if ddp:
        dist.barrier()  # 等主进程完成预测可视化后再统一退出
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
