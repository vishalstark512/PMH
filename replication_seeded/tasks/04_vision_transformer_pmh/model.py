"""Minimal ViT for CIFAR-32. Exposes block outputs for multi-scale PMH (last 3 blocks)."""
import math
import torch
import torch.nn as nn


def get_model(num_classes=10, img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0):
    return ViTCIFAR(
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, n_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        # x: (B, N, C)
        x = x + self._attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def _attn(self, x):
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class ViTCIFAR(nn.Module):
    """ViT for CIFAR-32. get_features(x, return_all=True) returns list of block outputs (CLS token) for PMH."""

    def __init__(self, num_classes=10, img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def get_features(self, x, return_all=False):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+n_patches, embed_dim)
        x = x + self.pos_embed
        features = []
        for blk in self.blocks:
            x = blk(x)
            features.append(x[:, 0])  # CLS token (B, embed_dim)
        if return_all:
            return features
        x = self.norm(x[:, 0])
        return x

    def forward(self, x, return_features=False):
        features = self.get_features(x, return_all=return_features)
        if return_features:
            x = self.norm(features[-1])
            logits = self.head(x)
            return logits, features
        logits = self.head(features)
        return logits
