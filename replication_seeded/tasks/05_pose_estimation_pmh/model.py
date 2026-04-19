"""
Image -> 3D pose (17 joints). Backbone + head; get_features() for PMH.
Replace backbone with ResNet/HRNet when using real images.
"""
import torch
import torch.nn as nn

NUM_JOINTS = 17


class PoseHead(nn.Module):
    """Maps feature vector to 17x3 pose (mm)."""
    def __init__(self, feat_dim=512, hidden=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, NUM_JOINTS * 3),
        )

    def forward(self, x):
        # x: (B, feat_dim)
        out = self.fc(x)
        return out.view(*out.shape[:-1], NUM_JOINTS, 3)


class ImageToPose(nn.Module):
    """Backbone + pose head. get_features(..., return_all=True) for PMH multi-scale."""
    def __init__(self, backbone="resnet18", pretrained=False):
        super().__init__()
        if backbone == "resnet18":
            from torchvision.models import resnet18
            res = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            self.stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
            self.layer1 = res.layer1
            self.layer2 = res.layer2
            self.layer3 = res.layer3
            self.layer4 = res.layer4
            feat_dim = 512
        else:
            raise ValueError(backbone)
        self.feat_dim = feat_dim
        self.head = PoseHead(feat_dim=feat_dim)

    def _forward_backbone(self, x):
        z = self.stem(x)
        z1 = self.layer1(z)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        return [z1, z2, z3, z4]

    def forward(self, x, return_features=False):
        features = self._forward_backbone(x)
        z_flat = features[-1].mean(dim=(2, 3))
        pose = self.head(z_flat)
        if return_features:
            return pose, features
        return pose

    def get_features(self, x, return_all=False):
        """Multi-scale features for PMH. return_all=True -> list of (B, C, h, w)."""
        features = self._forward_backbone(x)
        if return_all:
            return features
        return features[-1]


def get_model(backbone="resnet18", pretrained=False):
    return ImageToPose(backbone=backbone, pretrained=pretrained)
