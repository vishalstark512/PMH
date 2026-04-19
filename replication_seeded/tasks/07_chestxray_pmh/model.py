"""
Chest X-ray model: backbone (ResNet18) + embedding + multi-label classifier.
Same pattern as Re-ID; classifier has num_classes (14) outputs with BCE.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


def get_model(num_classes=14, embed_dim=512, pretrained=True):
    return ChestXrayNet(
        num_classes=num_classes,
        embed_dim=embed_dim,
        pretrained=pretrained,
    )


class ChestXrayNet(nn.Module):
    """
    ResNet18 backbone + global pool + embedding fc + multi-label classifier.
    forward(x) -> logits (B, num_classes)
    forward(x, return_embedding=True) -> (logits, embedding)
    forward(x, return_features=True) -> (logits, [f1..f4]) for PMH (Task 01–aligned)
    get_features(x, return_all=True) -> [layer1..layer4] maps; PMH uses last 3 scales.
    get_embedding(x) -> embedding (L2-normalized for eval)
    """

    def __init__(self, num_classes=14, embed_dim=512, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        base = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
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

    def get_features(self, x, return_all=False):
        """Return feature pyramid [f1,f2,f3,f4] (4D tensors) or pooled backbone vector if return_all=False."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        if return_all:
            return [f1, f2, f3, f4]
        x = self.avgpool(f4)
        return torch.flatten(x, 1)

    def _forward_backbone(self, x):
        return self.get_features(x, return_all=False)

    def get_stage_features(self, x):
        """Return list of (B, C) features after each ResNet stage (layer1..layer4), for mech interp."""
        feats_maps = self.get_features(x, return_all=True)
        return [self.avgpool(f).flatten(1) for f in feats_maps]

    def get_embedding(self, x, normalize=True):
        feat = self._forward_backbone(x)
        emb = self.embed(feat)
        if normalize:
            emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def forward(self, x, return_embedding=False, return_features=False):
        if return_features:
            features = self.get_features(x, return_all=True)
            pooled = self.avgpool(features[-1])
            pooled = torch.flatten(pooled, 1)
            emb = self.embed(pooled)
            logits = self.classifier(emb)
            return logits, features
        feat = self._forward_backbone(x)
        emb = self.embed(feat)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits
