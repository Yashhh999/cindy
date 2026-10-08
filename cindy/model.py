"""ViT-B/16 LoRA + fixed forensic filters.

Fits a 16 GB T4. Frozen OpenCLIP ViT-B/16 (LAION-2B) with rank-8 LoRA on
query and value, plus a small CNN on high-pass / SRM / FFT cues.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "cindy-vitb16-lora-freq-v1"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _mha_lora(attn, q_x, k_x, v_x, q_A, q_B, v_A, v_B, scale):
    """Self-attention with a LoRA residual on Q and V. x is LND or NLD."""
    batch_first = bool(getattr(attn, "batch_first", False))
    if attn.in_proj_weight is None:
        raise RuntimeError("OpenCLIP attention has no fused in_proj_weight")

    w = attn.in_proj_weight
    b = attn.in_proj_bias
    e = attn.embed_dim
    q = F.linear(q_x, w[:e], None if b is None else b[:e])
    k = F.linear(k_x, w[e : 2 * e], None if b is None else b[e : 2 * e])
    v = F.linear(v_x, w[2 * e :], None if b is None else b[2 * e :])

    def delta(src, A, B):
        # src @ A^T @ B^T, computed in fp32 so the low rank update is stable.
        d = src.float() @ A.float().t() @ B.float().t()
        return (d * scale).to(dtype=src.dtype)

    q = q + delta(q_x, q_A, q_B)
    v = v + delta(v_x, v_A, v_B)

    heads = attn.num_heads
    if batch_first:
        n, length, dim = q.shape
        hd = dim // heads
        def split(t):
            return t.view(n, length, heads, hd).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(split(q), split(k), split(v))
        out = out.permute(0, 2, 1, 3).contiguous().view(n, length, dim)
    else:
        length, n, dim = q.shape
        hd = dim // heads
        def split(t):
            return t.view(length, n, heads, hd).permute(1, 2, 0, 3)
        out = F.scaled_dot_product_attention(split(q), split(k), split(v))
        out = out.permute(2, 0, 1, 3).contiguous().view(length, n, dim)
    return F.linear(out, attn.out_proj.weight, attn.out_proj.bias)


def inject_lora(visual, rank=8, alpha=16):
    """Freeze the tower and add LoRA on every residual attention block."""
    visual.requires_grad_(False)
    if hasattr(visual, "patch_dropout") and hasattr(visual.patch_dropout, "prob"):
        visual.patch_dropout.prob = 0.0
    blocks = list(visual.transformer.resblocks)
    dim = blocks[0].attn.embed_dim
    scale = float(alpha) / float(rank)
    for block in blocks:
        q_A = nn.Parameter(torch.empty(rank, dim))
        q_B = nn.Parameter(torch.empty(dim, rank))
        v_A = nn.Parameter(torch.empty(rank, dim))
        v_B = nn.Parameter(torch.empty(dim, rank))
        nn.init.kaiming_uniform_(q_A, a=math.sqrt(5))
        nn.init.zeros_(q_B)
        nn.init.kaiming_uniform_(v_A, a=math.sqrt(5))
        nn.init.zeros_(v_B)
        block.lora_q_A = q_A
        block.lora_q_B = q_B
        block.lora_v_A = v_A
        block.lora_v_B = v_B
        block.lora_scale = scale
        orig = block.attention

        def attention(self, q_x, k_x=None, v_x=None, attn_mask=None, _orig=orig):
            if attn_mask is not None:
                return _orig(q_x, k_x, v_x, attn_mask)
            k_x = q_x if k_x is None else k_x
            v_x = q_x if v_x is None else v_x
            return _mha_lora(
                self.attn, q_x, k_x, v_x,
                self.lora_q_A, self.lora_q_B, self.lora_v_A, self.lora_v_B,
                self.lora_scale,
            )

        import types
        block.attention = types.MethodType(attention, block)
    return visual


def enable_grad_checkpoint(visual):
    transformer = visual.transformer
    import torch.utils.checkpoint as ckpt

    def forward(self, x, attn_mask=None):
        for block in self.resblocks:
            def run(tokens, module=block):
                return module(tokens)
            x = ckpt.checkpoint(run, x, use_reentrant=False)
        return x

    import types
    transformer.forward = types.MethodType(forward, transformer)


class FreqBranch(nn.Module):
    """Cheap cues that survive both old GANs and modern diffusion.

    Channels: RGB residual, 3 fixed SRM high-pass filters, FFT log-magnitude.
    """

    def __init__(self, out_dim=256):
        super().__init__()
        srm = torch.tensor(
            [
                [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
                [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
                [[-0.25, 0.5, -0.25], [0.5, -1.0, 0.5], [-0.25, 0.5, -0.25]],
            ],
            dtype=torch.float32,
        ).view(3, 1, 3, 3)
        self.register_buffer("srm", srm, persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(7, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        # x is RGB in [0, 1], not CLIP-normalized.
        blur = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        residual = x - blur
        gray = x.mean(dim=1, keepdim=True)
        srm = F.conv2d(gray, self.srm, padding=1)
        spec = torch.fft.fft2(gray)
        spec = torch.fft.fftshift(spec, dim=(-2, -1))
        mag = torch.log1p(spec.abs())
        mag = mag / mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        feat = torch.cat([residual, srm, mag], dim=1)
        h = self.net(feat).flatten(1)
        return self.proj(h)


class CindyDetector(nn.Module):
    def __init__(self, visual, freq_dim=256, grad_checkpoint=True):
        super().__init__()
        self.visual = visual
        if grad_checkpoint:
            enable_grad_checkpoint(self.visual)
        self.freq = FreqBranch(freq_dim)
        self.head = nn.Sequential(
            nn.Linear(768 + 768 + freq_dim, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def encode(self, x):
        v = self.visual
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(dtype=x.dtype, device=x.device)
        cls = cls.view(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = v.positional_embedding.to(dtype=x.dtype, device=x.device)
        if pos.shape[0] != x.shape[1]:
            raise RuntimeError(
                f"positional length {pos.shape[0]} != tokens {x.shape[1]}. Feed 224x224."
            )
        x = x + pos
        if hasattr(v, "patch_dropout"):
            x = v.patch_dropout(x)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = v.transformer(x)
        x = x.permute(1, 0, 2)
        cls = v.ln_post(x[:, :1]).squeeze(1)
        patches = v.ln_post(x[:, 1:]).mean(dim=1)
        return cls, patches

    def forward(self, images_01):
        freq = self.freq(images_01)
        normed = (images_01 - self.mean) / self.std
        cls, patches = self.encode(normed)
        return self.head(torch.cat([cls, patches, freq], dim=-1)).squeeze(-1)

    def trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        own = dict(self.named_parameters())
        missing = [k for k in state if k not in own]
        if missing:
            raise RuntimeError(f"checkpoint has unknown tensors: {missing[:8]}")
        with torch.no_grad():
            for name, tensor in state.items():
                own[name].copy_(tensor.to(device=own[name].device, dtype=own[name].dtype))


def build_model(rank0_log=print, lora_rank=8, lora_alpha=16, grad_checkpoint=True, pretrained="laion2b_s34b_b88k"):
    import open_clip

    rank0_log(f"loading OpenCLIP ViT-B-16 ({pretrained})")
    clip = open_clip.create_model("ViT-B-16", pretrained=pretrained)
    visual = clip.visual
    del clip
    if not hasattr(visual, "transformer"):
        raise RuntimeError("expected the classic OpenCLIP VisionTransformer, not a timm tower")
    inject_lora(visual, rank=lora_rank, alpha=lora_alpha)
    model = CindyDetector(visual, grad_checkpoint=grad_checkpoint)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    rank0_log(f"parameters total={n_all/1e6:.1f}M trainable={n_train/1e6:.2f}M")
    return model


def _self_test():
    torch.manual_seed(0)
    attn = nn.MultiheadAttention(32, 4, batch_first=False)
    q_A = torch.randn(4, 32) * 0.02
    q_B = torch.zeros(32, 4)
    v_A = torch.randn(4, 32) * 0.02
    v_B = torch.zeros(32, 4)
    x = torch.randn(8, 2, 32)
    y = _mha_lora(attn, x, x, x, q_A, q_B, v_A, v_B, 2.0)
    assert y.shape == x.shape, y.shape
    freq = FreqBranch(32)
    images = torch.rand(2, 3, 224, 224)
    z = freq(images)
    assert z.shape == (2, 32), z.shape
    print("self-test ok", tuple(y.shape), tuple(z.shape))


if __name__ == "__main__":
    _self_test()
