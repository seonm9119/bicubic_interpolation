import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_features=64, num_grow_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_grow_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features + num_grow_channels, num_grow_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_features + num_grow_channels * 2, num_grow_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_features + num_grow_channels * 3, num_grow_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_features + num_grow_channels * 4, num_features, 3, 1, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, image_features):
        feature1 = self.activation(self.conv1(image_features))
        feature2 = self.activation(self.conv2(torch.cat((image_features, feature1), dim=1)))
        feature3 = self.activation(self.conv3(torch.cat((image_features, feature1, feature2), dim=1)))
        feature4 = self.activation(self.conv4(torch.cat((image_features, feature1, feature2, feature3), dim=1)))
        feature5 = self.conv5(torch.cat((image_features, feature1, feature2, feature3, feature4), dim=1))

        return feature5 * 0.2 + image_features


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, num_features=64, num_grow_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_features, num_grow_channels)
        self.rdb2 = ResidualDenseBlock(num_features, num_grow_channels)
        self.rdb3 = ResidualDenseBlock(num_features, num_grow_channels)

    def forward(self, image_features):
        residual_features = self.rdb1(image_features)
        residual_features = self.rdb2(residual_features)
        residual_features = self.rdb3(residual_features)

        return residual_features * 0.2 + image_features


class RRDBNet(nn.Module):
    def __init__(self, scale=4, num_features=64, num_blocks=23, num_grow_channels=32):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(3, num_features, 3, 1, 1)
        self.body = nn.Sequential(*[
            ResidualInResidualDenseBlock(num_features, num_grow_channels)
            for _ in range(num_blocks)
        ])
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, 3, 3, 1, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, image_tensor):
        image_features = self.conv_first(image_tensor)
        body_features = self.conv_body(self.body(image_features))
        image_features = image_features + body_features
        image_features = self.activation(self.conv_up1(F.interpolate(image_features, scale_factor=2, mode="nearest")))
        image_features = self.activation(self.conv_up2(F.interpolate(image_features, scale_factor=2, mode="nearest")))
        output_features = self.conv_last(self.activation(self.conv_hr(image_features)))

        return output_features


def load_realesrgan_generator(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint
    realesrgan_generator = RRDBNet(scale=4)
    realesrgan_generator.load_state_dict(state_dict, strict=True)
    realesrgan_generator.to(device)
    realesrgan_generator.eval()

    return realesrgan_generator
