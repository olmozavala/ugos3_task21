import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from collections import OrderedDict


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "linear":
        return nn.Identity()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Activation {activation} not supported")

def low_pass_skip(skip_feature, cutoff_sigma=2.0):
    """
    Gaussian low-pass filter on skip connection feature maps.
    Blocks high-k noise from bypassing the bottleneck.
    cutoff_sigma: spatial scale of filtering in pixels.
    Larger = more aggressive smoothing.
    """
    kernel_size = int(4 * cutoff_sigma + 1)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    channels = skip_feature.shape[1]

    x = torch.arange(
        kernel_size,
        device=skip_feature.device,
        dtype=skip_feature.dtype,
    ) - kernel_size // 2
    gauss_1d = torch.exp(-x**2 / (2 * cutoff_sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]

    kernel = gauss_2d.expand(channels, 1, kernel_size, kernel_size)

    return F.conv2d(
        skip_feature,
        kernel,
        padding=kernel_size // 2,
        groups=channels  # depthwise: each channel independently
    )

class EncoderDecoderBlock(nn.Module):
    def __init__(self, level, cnn_per_level=2, activation="relu", in_filters=8, out_filters=16, kernel_size=3,
                 add_transpose=False, batch_norm=True):
        super().__init__()
        block_layers = OrderedDict()
        for i in range(cnn_per_level):
            layer_name = f'conv_l{level}_{i}'
            if i == 0:
                block_layers[layer_name] = nn.Conv2d(in_filters, out_filters, kernel_size, 1, padding='same')
            else:
                block_layers[layer_name] = nn.Conv2d(out_filters, out_filters, kernel_size, 1, padding='same')

            if batch_norm:
                layer_name = f'batch_l{level}_{i}'
                block_layers[layer_name] = nn.BatchNorm2d(out_filters)

            layer_name = f'act_l{level}_{i}'
            block_layers[layer_name] = get_activation(activation)

        if add_transpose:
            layer_name = f'upsample_l{level}'
            block_layers[layer_name] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            layer_name = f'conv_l{level}_up'
            block_layers[layer_name] = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1)
        self.block = nn.Sequential(block_layers)

    def forward(self, x):
        return self.block(x)

# Define the CNN architecture
class UNet(BaseModel):

    def __init__(self, previous_days, in_channels, start_filters=64, num_levels=4, 
                    kernel_size=3,  batch_norm=False, cnn_per_level=2, 
                    dropout_rate=0, hidden_activation="relu", output_activation="linear", dataset_type="regular"):
        """
        Initialize the UNet model.

        Args:
            previous_days (int): Number of previous days to consider for input.
            in_channels (int): Number of input channels for each day.
            out_channels (int): Number of output channels.
            start_filters (int, optional): Number of filters to start with. Defaults to 64.
            num_levels (int, optional): Number of levels in the U-Net architecture. Defaults to 4.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.
            out_activation (callable, optional): Activation function for the output layer. Defaults to None.
        """
        super(UNet, self).__init__()
        print("UNet model")
        cur_filters = -1  # Just initialize current filters

        # Derive the effective number of input channels expected by the first Conv2d.
        # This must stay in sync with SimSatelliteDataset.tot_inputs / __getitem__.
        if dataset_type == "gaussian_noise_only":
            # Two previous SSH snapshots with Gaussian noise + gulf mask.
            in_channels = 3
        elif dataset_type == "regular":
            in_channels = in_channels * previous_days + 1
        else:
            # e.g. "nemo_mdt", "gaussian_noise"
            in_channels = in_channels * previous_days + 3

        out_channels = 1

        encoder_blocks = OrderedDict()
        decoder_blocks = OrderedDict()
        self.maxpools = nn.ModuleList()
        self.levels = num_levels
        self.maxpool = nn.MaxPool2d(2, 2)
        # Simulated inputs (just to see how the dimensions get modified)
        cur_w = 648
        cur_h = 712
        print("---------- Encoder ----------")
        for c_level in range(1, num_levels):
            input_filters = in_channels if c_level == 1 else output_filters
            output_filters = start_filters if c_level == 1 else output_filters * 2
            print(f'Level {c_level} in: {input_filters}x{cur_w}x{cur_h}')
            c_encoder = EncoderDecoderBlock(c_level, cnn_per_level, hidden_activation, in_filters=input_filters,
                                            out_filters=output_filters, kernel_size=kernel_size, batch_norm=batch_norm)
            encoder_blocks[f'enc_lev_{c_level}'] = c_encoder
            cur_w = int((cur_w)/ 2)    # for padding = same
            cur_h = int((cur_h)/ 2)    # for padding = same
            print(f'-- Level {c_level} out: {output_filters}x{cur_w}x{cur_h}')
            cur_filters = output_filters
        # Decoder
        self.encoder_blocks = nn.Sequential(encoder_blocks)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in range(num_levels-1)])

        print("---------- Bottom ----------")
        input_filters = cur_filters   # Considering the skip connections
        print(f'-- Level Bottom in: {input_filters}x{cur_w}x{cur_h}')
        output_filters = int(cur_filters * 2)  # Considering the skip connections
        self.bottom = EncoderDecoderBlock('bottom', cnn_per_level, hidden_activation, in_filters=input_filters,
                                          out_filters=output_filters, kernel_size=kernel_size, batch_norm=batch_norm,
                                          add_transpose=True)
        cur_w = int((cur_w) * 2)  # for padding = same
        cur_h = int((cur_h) * 2)  # for padding = same
        print(f'-- Level Bottom out: {output_filters}x{cur_w}x{cur_h}')
        cur_filters = output_filters

        print("---------- Decoder ----------")
        for c_level in range(num_levels-1, 0, -1):
            input_filters = cur_filters + int(cur_filters/2)  # Considering the skip connections
            print(f'Level {c_level} in: {input_filters}(skip)x{cur_w}x{cur_h}')

            add_transpose = False if c_level == 1 else True
            output_filters = int(cur_filters/2) if add_transpose else input_filters

            decoder_blocks[f'dec_lev_{c_level}'] = EncoderDecoderBlock(c_level, cnn_per_level, hidden_activation, input_filters, output_filters,
                                                                       kernel_size, add_transpose=add_transpose, batch_norm=batch_norm)
            # cur_w = int((cur_w - 4) * 2) if add_transpose else cur_w - 4  # For padding = valid
            cur_w = int((cur_w) * 2) if add_transpose else cur_w  # For padding = same
            cur_h = int((cur_h) * 2) if add_transpose else cur_h  # For padding = same
            print(f'-- Level {c_level} out: {output_filters}x{cur_w}x{cur_h}')
            cur_filters = output_filters

        self.decoder_blocks = nn.Sequential(decoder_blocks)
        print(f'* Last layer in: {output_filters}x{cur_w}x{cur_h}')
        print(f'* Last layer out: {out_channels}x{cur_w}x{cur_h}')
        self.out_layer = EncoderDecoderBlock(c_level, 1, output_activation, output_filters, out_channels, kernel_size,
                                             add_transpose=add_transpose, batch_norm=batch_norm)

    def forward(self, x):
        encoder_outputs = {}
        for level, enc_block in enumerate(self.encoder_blocks):
            # print(f"Evaluating enc_{level+1}")
            # print(x.shape)
            x = enc_block(x)
            encoder_outputs[f'enc_{level+1}'] = x
            x = self.maxpools[level](x)
            # print(x.shape)

        # print(" ------- Bottom -------")
        # print(x.shape)
        x = self.bottom(x)
        # print(x.shape)

        # print(" ------- Decoder -------")
        for dec_level, dec_block in enumerate(self.decoder_blocks):
            # print(f"Evaluating dec_{dec_level+1}")
            dec_level_str = f'enc_{self.levels - dec_level - 1}'
            # print(f"Skip connection with {dec_level_str}")
            x = torch.cat((encoder_outputs[dec_level_str], x), dim=1)
            # print(x.shape)
            x = dec_block(x)
            # print(x.shape)

        return self.out_layer(x).squeeze(1)


class UNetNoSkip(BaseModel):
    """UNet without skip connections: encoder path is not concatenated into the decoder."""

    def __init__(self, previous_days, in_channels, start_filters=64, num_levels=4,
                 kernel_size=3, batch_norm=False, cnn_per_level=2,
                 dropout_rate=0, hidden_activation="relu", output_activation="linear", dataset_type="regular"):
        super(UNetNoSkip, self).__init__()
        print("UNetNoSkip model (no skip connections)")
        cur_filters = -1

        # Derive the effective number of input channels expected by the first Conv2d.
        if dataset_type == "gaussian_noise_only":
            in_channels = 3
        elif dataset_type == "regular":
            in_channels = in_channels * previous_days + 1
        else:
            in_channels = in_channels * previous_days + 3

        out_channels = 1

        encoder_blocks = OrderedDict()
        decoder_blocks = OrderedDict()
        self.maxpools = nn.ModuleList()
        self.levels = num_levels
        self.maxpool = nn.MaxPool2d(2, 2)
        cur_w = 648
        cur_h = 712
        print("---------- Encoder ----------")
        for c_level in range(1, num_levels):
            input_filters = in_channels if c_level == 1 else output_filters
            output_filters = start_filters if c_level == 1 else output_filters * 2
            print(f'Level {c_level} in: {input_filters}x{cur_w}x{cur_h}')
            c_encoder = EncoderDecoderBlock(c_level, cnn_per_level, hidden_activation, in_filters=input_filters,
                                            out_filters=output_filters, kernel_size=kernel_size, batch_norm=batch_norm)
            encoder_blocks[f'enc_lev_{c_level}'] = c_encoder
            cur_w = int((cur_w) / 2)
            cur_h = int((cur_h) / 2)
            print(f'-- Level {c_level} out: {output_filters}x{cur_w}x{cur_h}')
            cur_filters = output_filters

        self.encoder_blocks = nn.Sequential(encoder_blocks)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in range(num_levels - 1)])

        print("---------- Bottom ----------")
        input_filters = cur_filters
        print(f'-- Level Bottom in: {input_filters}x{cur_w}x{cur_h}')
        output_filters = int(cur_filters * 2)
        self.bottom = EncoderDecoderBlock('bottom', cnn_per_level, hidden_activation, in_filters=input_filters,
                                          out_filters=output_filters, kernel_size=kernel_size, batch_norm=batch_norm,
                                          add_transpose=True)
        cur_w = int((cur_w) * 2)
        cur_h = int((cur_h) * 2)
        print(f'-- Level Bottom out: {output_filters}x{cur_w}x{cur_h}')
        cur_filters = output_filters

        print("---------- Decoder (no skip) ----------")
        for c_level in range(num_levels - 1, 0, -1):
            input_filters = cur_filters  # No skip: decoder input = current filters only
            print(f'Level {c_level} in: {input_filters}x{cur_w}x{cur_h}')

            add_transpose = False if c_level == 1 else True
            output_filters = int(cur_filters / 2) if add_transpose else input_filters

            decoder_blocks[f'dec_lev_{c_level}'] = EncoderDecoderBlock(c_level, cnn_per_level, hidden_activation,
                                                                        input_filters, output_filters,
                                                                        kernel_size, add_transpose=add_transpose,
                                                                        batch_norm=batch_norm)
            cur_w = int((cur_w) * 2) if add_transpose else cur_w
            cur_h = int((cur_h) * 2) if add_transpose else cur_h
            print(f'-- Level {c_level} out: {output_filters}x{cur_w}x{cur_h}')
            cur_filters = output_filters

        self.decoder_blocks = nn.Sequential(decoder_blocks)
        print(f'* Last layer in: {output_filters}x{cur_w}x{cur_h}')
        print(f'* Last layer out: {out_channels}x{cur_w}x{cur_h}')
        self.out_layer = EncoderDecoderBlock(c_level, 1, output_activation, output_filters, out_channels, kernel_size,
                                              add_transpose=add_transpose, batch_norm=batch_norm)

    def forward(self, x):
        for level, enc_block in enumerate(self.encoder_blocks):
            x = enc_block(x)
            x = self.maxpools[level](x)

        x = self.bottom(x)

        for dec_block in self.decoder_blocks:
            x = dec_block(x)

        return self.out_layer(x).squeeze(1)


# Define the CNN architecture with low-pass filter on skip connections
class UNetLowPass(BaseModel):

    def __init__(self, previous_days, in_channels, start_filters=64, num_levels=4, 
                    kernel_size=3,  batch_norm=False, cnn_per_level=2, 
                    dropout_rate=0, hidden_activation="relu", output_activation="linear", dataset_type="regular"):
        """
        Initialize the UNet model.

        Args:
            previous_days (int): Number of previous days to consider for input.
            in_channels (int): Number of input channels for each day.
            out_channels (int): Number of output channels.
            start_filters (int, optional): Number of filters to start with. Defaults to 64.
            num_levels (int, optional): Number of levels in the U-Net architecture. Defaults to 4.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.
            out_activation (callable, optional): Activation function for the output layer. Defaults to None.
        """
        super(UNetLowPass, self).__init__()
        print("UNetLowPass model")
        cur_filters = -1  # Just initialize current filters

        # Derive the effective number of input channels expected by the first Conv2d.
        # This must stay in sync with SimSatelliteDataset.tot_inputs / __getitem__.
        if dataset_type == "gaussian_noise_only":
            # Two previous SSH snapshots with Gaussian noise + gulf mask.
            in_channels = 3
        elif dataset_type == "regular":
            in_channels = in_channels * previous_days + 1
        else:
            # e.g. "nemo_mdt", "gaussian_noise"
            in_channels = in_channels * previous_days + 3

        out_channels = 1

        encoder_blocks = OrderedDict()
        decoder_blocks = OrderedDict()
        self.maxpools = nn.ModuleList()
        self.levels = num_levels
        self.maxpool = nn.MaxPool2d(2, 2)
        # Simulated inputs (just to see how the dimensions get modified)
        cur_w = 648
        cur_h = 712
        print("---------- Encoder ----------")
        for c_level in range(1, num_levels):
            input_filters = in_channels if c_level == 1 else output_filters
            output_filters = start_filters if c_level == 1 else output_filters * 2
            print(f'Level {c_level} in: {input_filters}x{cur_w}x{cur_h}')
            c_encoder = EncoderDecoderBlock(c_level, cnn_per_level, hidden_activation, in_filters=input_filters,
                                            out_filters=output_filters, kernel_size=kernel_size, batch_norm=batch_norm)
            encoder_blocks[f'enc_lev_{c_level}'] = c_encoder
            cur_w = int((cur_w)/ 2)    # for padding = same
            cur_h = int((cur_h)/ 2)    # for padding = same
            print(f'-- Level {c_level} out: {output_filters}x{cur_w}x{cur_h}')
            cur_filters = output_filters
        # Decoder
        self.encoder_blocks = nn.Sequential(encoder_blocks)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in range(num_levels-1)])

        print("---------- Bottom ----------")
        input_filters = cur_filters   # Considering the skip connections
        print(f'-- Level Bottom in: {input_filters}x{cur_w}x{cur_h}')
        output_filters = int(cur_filters * 2)  # Considering the skip connections
        self.bottom = EncoderDecoderBlock('bottom', cnn_per_level, hidden_activation, in_filters=input_filters,
                                          out_filters=output_filters, kernel_size=kernel_size, batch_norm=batch_norm,
                                          add_transpose=True)
        cur_w = int((cur_w) * 2)  # for padding = same
        cur_h = int((cur_h) * 2)  # for padding = same
        print(f'-- Level Bottom out: {output_filters}x{cur_w}x{cur_h}')
        cur_filters = output_filters

        print("---------- Decoder ----------")
        for c_level in range(num_levels-1, 0, -1):
            input_filters = cur_filters + int(cur_filters/2)  # Considering the skip connections
            print(f'Level {c_level} in: {input_filters}(skip)x{cur_w}x{cur_h}')

            add_transpose = False if c_level == 1 else True
            output_filters = int(cur_filters/2) if add_transpose else input_filters

            decoder_blocks[f'dec_lev_{c_level}'] = EncoderDecoderBlock(c_level, cnn_per_level, hidden_activation, input_filters, output_filters,
                                                                       kernel_size, add_transpose=add_transpose, batch_norm=batch_norm)
            # cur_w = int((cur_w - 4) * 2) if add_transpose else cur_w - 4  # For padding = valid
            cur_w = int((cur_w) * 2) if add_transpose else cur_w  # For padding = same
            cur_h = int((cur_h) * 2) if add_transpose else cur_h  # For padding = same
            print(f'-- Level {c_level} out: {output_filters}x{cur_w}x{cur_h}')
            cur_filters = output_filters

        self.decoder_blocks = nn.Sequential(decoder_blocks)
        print(f'* Last layer in: {output_filters}x{cur_w}x{cur_h}')
        print(f'* Last layer out: {out_channels}x{cur_w}x{cur_h}')
        self.out_layer = EncoderDecoderBlock(c_level, 1, output_activation, output_filters, out_channels, kernel_size,
                                             add_transpose=add_transpose, batch_norm=batch_norm)

    def forward(self, x):
        encoder_outputs = {}
        for level, enc_block in enumerate(self.encoder_blocks):
            # print(f"Evaluating enc_{level+1}")
            # print(x.shape)
            x = enc_block(x)
            encoder_outputs[f'enc_{level+1}'] = x
            x = self.maxpools[level](x)
            # print(x.shape)

        # print(" ------- Bottom -------")
        # print(x.shape)
        x = self.bottom(x)
        # print(x.shape)

        # print(" ------- Decoder -------")
        for dec_level, dec_block in enumerate(self.decoder_blocks):
            # print(f"Evaluating dec_{dec_level+1}")
            dec_level_str = f'enc_{self.levels - dec_level - 1}'
            skip_filtered = low_pass_skip(encoder_outputs[dec_level_str], cutoff_sigma=2.0)
            # print(f"Skip connection with {dec_level_str}")
            x = torch.cat((skip_filtered, x), dim=1)
            # print(x.shape)
            x = dec_block(x)
            # print(x.shape)

        return self.out_layer(x).squeeze(1)