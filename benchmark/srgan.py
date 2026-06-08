import torch
from torch import nn


def create_same_padding_layer(kernel_size, mode="reflection"):
    total_padding = kernel_size - 1
    padding_side = total_padding // 2

    if total_padding % 2 == 0:
        padding = padding_side
    else:
        padding = (padding_side, padding_side + 1, padding_side, padding_side + 1)

    if mode == "reflection":
        return nn.ReflectionPad2d(padding)
    if mode == "replication":
        return nn.ReplicationPad2d(padding)

    return nn.ZeroPad2d(padding)


class ResBlock(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size=3, padding="reflection"):
        super().__init__()
        self.block = nn.Sequential(
            create_same_padding_layer(kernel_size, padding),
            nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.PReLU(num_parameters=num_filters, init=0.0),
            create_same_padding_layer(kernel_size, padding),
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(num_filters, affine=True),
        )

    def forward(self, image_tensor):
        return self.block(image_tensor) + image_tensor


class SRResNet(nn.Module):
    def __init__(self, upscale_factor=4, num_inputs=3, num_outputs=3, num_filters=64, num_res_blocks=16, padding="reflection"):
        super().__init__()
        self.initial_conv = nn.Sequential(
            create_same_padding_layer(9, padding),
            nn.Conv2d(num_inputs, num_filters, kernel_size=9, stride=1, bias=True),
            nn.PReLU(num_parameters=num_filters, init=0.0),
        )

        residual_blocks = [
            ResBlock(num_filters, num_filters, kernel_size=3, padding=padding)
            for _ in range(num_res_blocks)
        ]
        residual_blocks.extend([
            create_same_padding_layer(3, padding),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(num_filters, affine=True),
        ])
        self.body = nn.Sequential(*residual_blocks)

        upsample_layers = []
        in_channels = num_filters
        scale = 2 if upscale_factor % 2 == 0 else 3

        for _ in range(upscale_factor // scale):
            upsample_layers.extend([
                create_same_padding_layer(3, padding),
                nn.Conv2d(in_channels, scale * scale * 256, kernel_size=3, stride=1, bias=True),
                nn.PixelShuffle(upscale_factor=scale),
                nn.PReLU(num_parameters=256, init=0.0),
            ])
            in_channels = 256

        self.upsample = nn.Sequential(*upsample_layers)
        self.final_conv = nn.Sequential(
            create_same_padding_layer(9, padding),
            nn.Conv2d(in_channels, num_outputs, kernel_size=9, stride=1, bias=True),
        )

    def forward(self, image_tensor):
        initial_features = self.initial_conv(image_tensor)
        residual_features = self.body(initial_features)
        upsampled_features = self.upsample(residual_features + initial_features)

        return self.final_conv(upsampled_features)


def load_srgan_generator(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator_state = checkpoint["runner"]["generator"]
    srgan_generator = SRResNet(upscale_factor=4)
    srgan_generator.load_state_dict(generator_state)
    srgan_generator.to(device)
    srgan_generator.eval()

    return srgan_generator
