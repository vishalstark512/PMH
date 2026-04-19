"""CIFAR-10 models: ResNet18CIFAR (default, matches notebook) and CIFAR10CNN (optional)."""
import torch
import torch.nn as nn
from torchvision.models import resnet18

# Default model for all runs (B0, B1, E1) — matches RES/CLASS notebook.
def get_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        return ResNet18CIFAR(num_classes=num_classes)
    if name == "cnn":
        return CIFAR10CNN(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-32 (matches notebook). Exposes multi-scale features for PMH.
    get_features(x, return_all=True) returns [layer1, layer2, layer3, layer4]; PMH uses last 3.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None, num_classes=num_classes)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc

    def get_features(self, x, return_all=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        if return_all:
            return [f1, f2, f3, f4]
        x = self.avgpool(f4)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, return_features=False):
        if return_features:
            features = self.get_features(x, return_all=True)
            x = self.avgpool(features[-1])
            x = torch.flatten(x, 1)
            logits = self.fc(x)
            return logits, features
        x = self.get_features(x, return_all=False)
        return self.fc(x)


class CIFAR10CNN(nn.Module):
    """Small CNN for CIFAR-10. Returns (logits, [feats]) when return_features=True."""
    def __init__(self, num_classes=10, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, return_features=False):
        out = self.conv(x)
        feats = self.feat_fc(out)
        logits = self.classifier(feats)
        if return_features:
            return logits, [feats]
        return logits
