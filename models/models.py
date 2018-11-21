import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, resnet101


class DetectionModel(nn.Module):
    """

    """

    def __init__(self, base_model=resnet101, num_templates=1, num_objects=1):
        super().__init__()
        output = (num_objects + 4)*num_templates  # 4 is for the bounding box offsets
        self.model = base_model(pretrained=True)
        self.score_res3 = nn.Conv2d(in_channels=512, out_channels=output, kernel_size=1)
        self.score_res4 = nn.Conv2d(in_channels=1024, out_channels=output, kernel_size=1)

    #     self.score4 = nn.ConvTranspose2d(in_channels=output, out_channels=output, kernel_size=4, stride=2, padding=1)
    #     self._init_bilinear()
    #
    # def _init_bilinear(self):
    #     """
    #     Initialize the ConvTranspose2d layer with a bilinear interpolation mapping
    #     :return:
    #     """
    #     k = self.score4.kernel_size[0]
    #     factor = np.floor((k+1)/2)
    #     if k % 2 == 1:
    #         center = factor
    #     else:
    #         center = factor + 0.5
    #     C = np.arange(1, 5)
    #
    #     f = np.zeros((self.score4.in_channels, self.score4.out_channels, k, k))
    #
    #     for i in range(self.score4.out_channels):
    #         f[i, i, :, :] = (np.ones((1, k)) - (np.abs(C-center)/factor)).T @ \
    #                         (np.ones((1, k)) - (np.abs(C-center)/factor))
    #
    #     self.score4.weight = torch.nn.Parameter(data=torch.Tensor(f))

    def learnable_parameters(self, lr):
        parameters = [
            {'params': self.model.parameters(), 'lr': lr},  # Be T'Challa. Don't freeze.
            {'params': self.score_res3.parameters(), 'lr': 0.1*lr},
            {'params': self.score_res4.parameters(), 'lr': 1*lr},
            # {'params': self.score4.parameters(), 'lr': 0}  # freeze UpConv layer
        ]
        return parameters

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        res2 = x

        x = self.model.layer2(x)
        res3 = x

        x = self.model.layer3(x)
        res4 = x

        x = self.model.layer4(x)
        res5 = x

        score_res3 = self.score_res3(res3)

        score_res4 = self.score_res4(res4)
        # score4 = self.score4(score_res4)
        score4 = nn.functional.interpolate(score_res4, size=score_res3.shape[2:], mode='bilinear', align_corners=False)

        # if not self.training:  # then we need to do some fancy cropping to accomodate the difference in image sizes
        #     # from vl_feats DagNN Crop
        #     cropv = score4.size(2) - score_res3.size(2)
        #     cropu = score4.size(3) - score_res3.size(3)
        #     # if the crop is 0 (both the input sizes are the same)
        #     # we do some arithmetic to allow python to index correctly
        #     if cropv == 0:
        #         cropv = -score4.size(2)
        #     if cropu == 0:
        #         cropu = -score4.size(3)
        #
        #     score4 = score4[:, :, 0:-cropv, 0:-cropu]

        score = score_res3 + score4

        return score

