import math
from torch import nn
import torch.utils.model_zoo as model_zoo



class HyperColumn(nn.Module):
    """
    Model to extract features as HyperColumns
    """

    def __init__(self, upsample_mode='bilinear', output=5):
        super().__init__()

        self.score_res2 = nn.Conv2d(in_channels=256, out_channels=output, kernel_size=1)
        self.score_res3 = nn.Conv2d(in_channels=512, out_channels=output, kernel_size=1)
        self.score_res4 = nn.Conv2d(in_channels=1024, out_channels=output, kernel_size=1)
        self.score_res5 = nn.Conv2d(in_channels=2048, out_channels=output, kernel_size=1)

        self.upsample_mode = upsample_mode

    def upsample(self, x, shape):
        return nn.functional.interpolate(x, size=shape, mode=self.upsample_mode, align_corners=False)

    def forward(self, x):
        res2, res3, res4, res5 = x

        score_res5 = self.score_res5(res5)
        score_res4 = self.score_res4(res4)
        score_res3 = self.score_res3(res3)
        score_res2 = self.score_res2(res2)

        upsample_shape = score_res3.shape[2:]

        score5 = self.upsample(score_res5, upsample_shape)
        score4 = self.upsample(score_res4, upsample_shape)
        score3 = self.upsample(score_res3, upsample_shape)
        score2 = self.upsample(score_res2, upsample_shape)

        score = score2 + score3 + score4 + score5
        # score = score3 + score4
        return score


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HiResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(HiResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        res2 = x
        x = self.layer2(x)
        res3 = x
        x = self.layer3(x)
        res4 = x
        x = self.layer4(x)
        res5 = x

        return res2, res3, res4, res5


def hiresnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HiResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

    # delete FC layer to save space
    del model.fc
    return model


class DetectionModel(nn.Module):
    """
    Fast Detection Model using highest resolution map of Feature Pyramid Network
    """

    def __init__(self, feature_extractor="HyperColumn", num_objects=1, num_templates=1, num_scales=1):
        super().__init__()
        output = (num_objects + 4) * num_templates  # 4 is for the bounding box offsets

        # the number of scales we wish to train at.
        self.num_scales = num_scales

        self.extractor_type = feature_extractor
        if feature_extractor == "HyperColumn":
            self.feature_extractor = HyperColumn(output=output)

        # Final classifier
        self.score = nn.Conv2d(in_channels=256, out_channels=output, kernel_size=3, padding=1)

    def learnable_parameters(self, lr, d=1):
        parameters = [
            {'params': self.feature_extractor.parameters(), 'lr': lr},  # Be T'Challa. Don't freeze.
            {'params': self.score.parameters(), 'lr': d*lr},
        ]
        return parameters

    def forward(self, x, mask=None):
        feat = self.feature_extractor(x)

        score = None
        if self.extractor_type == "FPN":
            score = self.score(feat)
        elif self.extractor_type == "HyperColumn":
            score = feat

        return score
