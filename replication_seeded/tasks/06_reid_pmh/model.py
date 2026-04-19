"""
Re-ID model: backbone (ResNet18) + embedding layer + ID classifier.
Exposes embedding for metric learning and PMH (same identity → same φ).
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


def get_model(num_classes, embed_dim=512, pretrained=True):
    return ReIDNet(
        num_classes=num_classes,
        embed_dim=embed_dim,
        pretrained=pretrained,
    )


class ReIDNet(nn.Module):
    """
    ResNet18 backbone + global pool + embedding fc + classifier.
    forward(x) -> logits
    forward(x, return_embedding=True) -> (logits, embedding)
    get_embedding(x) -> embedding (L2-normalized for eval)
    """

    def __init__(self, num_classes, embed_dim=512, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        base = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # Remove original fc; keep backbone
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        feat_dim = 512

        self.embed = nn.Linear(feat_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _forward_backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_embedding(self, x, normalize=True):
        """Return embedding (B, embed_dim). Optionally L2-normalized."""
        feat = self._forward_backbone(x)
        emb = self.embed(feat)
        if normalize:
            emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def get_features(self, x, return_all=False):
        """Return intermediate stage features for multi-scale PMH.
        Returns list: [pool(layer2), pool(layer3), pool(layer4)] as flat vectors (before embed)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        pool = lambda t: torch.flatten(self.avgpool(t), 1)
        return [pool(f2), pool(f3), pool(f4)]

    def forward(self, x, return_embedding=False, return_features=False):
        x_ = self.conv1(x); x_ = self.bn1(x_); x_ = self.relu(x_); x_ = self.maxpool(x_)
        x_ = self.layer1(x_)
        f2 = self.layer2(x_); f3 = self.layer3(f2); f4 = self.layer4(f3)
        pool = lambda t: torch.flatten(self.avgpool(t), 1)
        feat = pool(f4)
        emb = self.embed(feat)
        logits = self.classifier(emb)
        if return_features:
            return logits, emb, [pool(f2), pool(f3), feat]
        if return_embedding:
            return logits, emb
        return logits
