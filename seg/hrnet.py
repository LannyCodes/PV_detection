"""
hrnet.py — HRNet 语义分割模型定义 (W18 / W32 / W48)

精简移植自官方实现 "HRNet-Semantic-Segmentation"
(论文: Deep High-Resolution Representation Learning for Visual Recognition,
 CVPR 2019 / TPAMI 2020): 仅保留语义分割所需的最小结构, 单文件自包含,
不依赖 mmcv / mmsegmentation, 在 Kaggle 上零编译安装负担。

结构概览:
    stem (2× stride-2 卷积, 1/4 分辨率)
      └─ layer1  高分辨率单分支
           └─ stage2 (2 分支并行) ← 分支间反复交换特征
                └─ stage3 (3 分支)
                     └─ stage4 (4 分支)
                          └─ HRNetV2 头: 各分支上采样至 1/4 分辨率 → 拼接
                               → 1×1 卷积融合 → 1×1 卷积输出类别 logits
                               → 双线性上采样回原图尺寸

用法:
    from hrnet import build_hrnet
    model = build_hrnet("w18", num_classes=2)   # 板面分割: 0=背景, 1=光伏板
    logits = model(images)                       # (B, 2, H, W), 推理时 argmax 得掩码

自检 (打印参数量并验证前向):
    python seg/hrnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_hrnet", "HRNetSegmentation"]

# 官方语义分割配置: 每个 stage 的块数均为 4, 全部使用 BasicBlock
# 参数量参考: W18 ≈ 9.6M / W32 ≈ 29M / W48 ≈ 63M
ARCH_CONFIGS = {
    "w18": {
        "stem_width": 64,
        "stage1": {"num_blocks": [4], "num_channels": [64]},
        "stage2": {"num_modules": 1, "num_blocks": [4, 4], "num_channels": [18, 36]},
        "stage3": {"num_modules": 4, "num_blocks": [4, 4, 4], "num_channels": [18, 36, 72]},
        "stage4": {"num_modules": 3, "num_blocks": [4, 4, 4, 4], "num_channels": [18, 36, 72, 144]},
    },
    "w32": {
        "stem_width": 64,
        "stage1": {"num_blocks": [4], "num_channels": [64]},
        "stage2": {"num_modules": 1, "num_blocks": [4, 4], "num_channels": [32, 64]},
        "stage3": {"num_modules": 4, "num_blocks": [4, 4, 4], "num_channels": [32, 64, 128]},
        "stage4": {"num_modules": 3, "num_blocks": [4, 4, 4, 4], "num_channels": [32, 64, 128, 256]},
    },
    "w48": {
        "stem_width": 64,
        "stage1": {"num_blocks": [4], "num_channels": [64]},
        "stage2": {"num_modules": 1, "num_blocks": [4, 4], "num_channels": [48, 96]},
        "stage3": {"num_modules": 4, "num_blocks": [4, 4, 4], "num_channels": [48, 96, 192]},
        "stage4": {"num_modules": 3, "num_blocks": [4, 4, 4, 4], "num_channels": [48, 96, 192, 384]},
    },
}


def conv3x3(inplanes, planes, stride=1):
    return nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)


def conv1x1(inplanes, planes, stride=1):
    return nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """基础残差块 (2 层 3×3 卷积), HRNet 官方各宽度均使用此块"""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """瓶颈残差块 (1×1 → 3×3 → 1×1), 供自定义大宽度变体使用"""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


_BLOCKS = {"basic": BasicBlock, "bottleneck": Bottleneck}


def make_layer(block, inplanes, planes, blocks, stride=1):
    """堆叠 blocks 个残差块 (首块带 stride 与 downsample 通道对齐)"""
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


class HighResolutionModule(nn.Module):
    """多分支并行 + 跨分支特征交换 (HRNet 论文核心模块)

    x_list 各分支分辨率依次减半 (1/4, 1/8, 1/16, 1/32), 每轮先过各分支的
    残差块, 再沿 (j→i) 方向对齐分辨率与通道后相加融合, 最后统一过 ReLU。
    """

    def __init__(self, block, num_blocks, num_channels):
        super().__init__()
        self.num_branches = len(num_channels)
        self.num_channels = num_channels
        self.branches = nn.ModuleList(
            self._make_branch(block, num_blocks[i], num_channels[i])
            for i in range(self.num_branches)
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _make_branch(block, num_blocks, channels):
        layers = [block(channels, channels)]
        for _ in range(1, num_blocks):
            layers.append(block(channels, channels))
        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        """构建跨分支对齐层: 低→高分辨率用 1×1 卷积(上采样在 forward 做), 高→低用 stride-2 3×3 卷积"""
        if self.num_branches == 1:
            return None
        ch = self.num_channels
        fuse_layers = []
        for i in range(self.num_branches):
            layer = []
            for j in range(self.num_branches):
                if j > i:
                    # x[j] 分辨率更低: 先做通道对齐, 上采样在 forward 中按目标尺寸做
                    layer.append(nn.Sequential(
                        conv1x1(ch[j], ch[i]),
                        nn.BatchNorm2d(ch[i]),
                    ))
                elif j == i:
                    layer.append(None)
                else:
                    # j < i, x[j] 分辨率更高: 逐级 stride-2 下采样
                    convs = []
                    for k in range(i - j):
                        last = k == i - j - 1
                        out_ch = ch[i] if last else ch[j]
                        seq = [conv3x3(ch[j], out_ch, stride=2), nn.BatchNorm2d(out_ch)]
                        if not last:
                            seq.append(nn.ReLU(inplace=True))
                        convs.append(nn.Sequential(*seq))
                    layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x_list):
        if self.num_branches == 1:
            return [self.branches[0](x_list[0])]

        x_list = [self.branches[i](x_list[i]) for i in range(self.num_branches)]
        out = []
        for i in range(self.num_branches):
            y = x_list[0] if i == 0 else self.fuse_layers[i][0](x_list[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x_list[j]
                elif j > i:
                    # 低分辨率 → 高分辨率: 上采样对齐后相加
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x_list[j]),
                        size=x_list[i].shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                else:
                    # 高分辨率 → 低分辨率: stride-2 卷积对齐后相加
                    y = y + self.fuse_layers[i][j](x_list[j])
            out.append(self.relu(y))
        return out


class HRNetSegmentation(nn.Module):
    """HRNet 语义分割网络 (HRNetV2 头), 输出与输入同分辨率的类别 logits"""

    def __init__(self, config, num_classes):
        super().__init__()
        stem_w = config["stem_width"]
        s1 = config["stage1"]

        # stem: 2 次 stride-2 卷积 → 1/4 分辨率
        self.conv1 = conv3x3(3, stem_w, stride=2)
        self.bn1 = nn.BatchNorm2d(stem_w)
        self.conv2 = conv3x3(stem_w, stem_w, stride=2)
        self.bn2 = nn.BatchNorm2d(stem_w)
        self.relu = nn.ReLU(inplace=True)

        # stage1: 高分辨率单分支
        self.layer1 = make_layer(BasicBlock, stem_w, s1["num_channels"][0], s1["num_blocks"][0])

        # stage2 ~ stage4: 分支逐级增多 (2 → 3 → 4), 分支间经 transition 交换
        self.transition1 = self._make_transition(s1["num_channels"], config["stage2"]["num_channels"])
        self.stage2 = self._make_stage(config["stage2"])
        self.transition2 = self._make_transition(config["stage2"]["num_channels"], config["stage3"]["num_channels"])
        self.stage3 = self._make_stage(config["stage3"])
        self.transition3 = self._make_transition(config["stage3"]["num_channels"], config["stage4"]["num_channels"])
        self.stage4 = self._make_stage(config["stage4"])

        # HRNetV2 头: 4 分支统一到 1/4 分辨率拼接融合 → 类别 logits
        head_ch = sum(config["stage4"]["num_channels"])
        self.head_fuse = nn.Sequential(
            conv1x1(head_ch, head_ch),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.head_classifier = nn.Conv2d(head_ch, num_classes, 1)

        self._init_weights()

    @staticmethod
    def _make_stage(cfg):
        block = _BLOCKS[cfg.get("block", "basic")]
        return nn.ModuleList(
            HighResolutionModule(block, cfg["num_blocks"], cfg["num_channels"])
            for _ in range(cfg["num_modules"])
        )

    @staticmethod
    def _make_transition(pre_channels, cur_channels):
        """stage 过渡层: 已有分支通道对齐 (相同则 None); 新增分支由上一 stage 最低分辨率分支 stride-2 得到"""
        pre_n = len(pre_channels)
        layers = []
        for i, cur_ch in enumerate(cur_channels):
            if i < pre_n:
                if cur_ch == pre_channels[i]:
                    layers.append(None)
                else:
                    layers.append(nn.Sequential(
                        conv3x3(pre_channels[i], cur_ch),
                        nn.BatchNorm2d(cur_ch),
                        nn.ReLU(inplace=True),
                    ))
            else:
                convs = []
                in_ch = pre_channels[-1]
                steps = i - pre_n + 1
                for k in range(steps):
                    out_ch = cur_ch if k == steps - 1 else in_ch
                    convs.append(nn.Sequential(
                        conv3x3(in_ch, out_ch, stride=2),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    ))
                    in_ch = out_ch
                layers.append(nn.Sequential(*convs))
        return nn.ModuleList(layers)

    @staticmethod
    def _run_transition(transition, x_list):
        out = []
        for i, layer in enumerate(transition):
            if i < len(x_list):
                out.append(layer(x_list[i]) if layer is not None else x_list[i])
            else:
                # 新增分支: 输入为上一 stage 的最低分辨率分支
                out.append(layer(x_list[-1]))
        return out

    @staticmethod
    def _run_stage(modules, x_list):
        for m in modules:
            x_list = m(x_list)
        return x_list

    def _init_weights(self):
        """官方初始化: 卷积 normal(std=0.001), BN 恒等初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h, w = x.shape[-2:]
        # 输入 pad 到 32 的倍数 (4 级 stride-2 累计), 避免多分支下采样尺寸不齐
        pad_h, pad_w = (32 - h % 32) % 32, (32 - w % 32) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)  # 1/4 分辨率

        x_list = self._run_transition(self.transition1, [x])
        x_list = self._run_stage(self.stage2, x_list)
        x_list = self._run_transition(self.transition2, x_list)
        x_list = self._run_stage(self.stage3, x_list)
        x_list = self._run_transition(self.transition3, x_list)
        x_list = self._run_stage(self.stage4, x_list)  # [1/4, 1/8, 1/16, 1/32]

        # HRNetV2 头: 统一到 1/4 分辨率后拼接融合
        target_hw = x_list[0].shape[-2:]
        feats = [x_list[0]]
        for feat in x_list[1:]:
            if feat.shape[-2:] != target_hw:
                feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
            feats.append(feat)
        x = self.head_classifier(self.head_fuse(torch.cat(feats, dim=1)))

        # 上采样回原图尺寸并裁掉 padding
        x = F.interpolate(x, size=(h + pad_h, w + pad_w), mode="bilinear", align_corners=False)
        if pad_h or pad_w:
            x = x[..., :h, :w]
        return x


def build_hrnet(arch="w18", num_classes=2):
    """构建 HRNet 分割模型

    Args:
        arch: "w18" / "w32" / "w48" (兼容 "hrnet_w18" / "HRNetV2-W18" 等写法)
        num_classes: 输出类别数 (含背景)
    """
    key = arch.lower()
    for cand in ARCH_CONFIGS:
        if cand in key:
            key = cand
            break
    if key not in ARCH_CONFIGS:
        raise ValueError(f"未知架构 {arch!r}, 可选: {sorted(ARCH_CONFIGS)}")
    return HRNetSegmentation(ARCH_CONFIGS[key], num_classes)


if __name__ == "__main__":
    # 快速自检: 参数量与前向输出形状 (随机输入, 验证结构连通性)
    for arch in ("w18", "w32"):
        model = build_hrnet(arch, num_classes=2)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        dummy = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out = model(dummy)
        print(f"HRNet-{arch.upper()}: {n_params:.1f}M 参数 | 输入 {tuple(dummy.shape)} → 输出 {tuple(out.shape)}")
