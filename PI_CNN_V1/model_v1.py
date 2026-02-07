# this is the my model that tries to implement partial convolutions and 
# first we need to do double convolutions 

import torch 
import torch.nn as nn
import torch.nn.functional as F

#double convolutions

class decoder_double_convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_double_convolution, self).__init__()

        self.convolution_operation = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                  out_channels=out_channels,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(in_channels=out_channels, 
                  out_channels=out_channels,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2,inplace=False),
        )
    def forward(self, x):
        return self.convolution_operation(x)
        
class decoder_final_layer_convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_final_layer_convolution, self).__init__()

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels,
                      kernel_size=1, 
                      stride=1,
                      padding=0),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.final_conv(x)

# partial convolutions class 
class partial_convolutions(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,padding, negative_slope, bias = False):
        #use the analogy that we're going to go to the deli and getting a sandwich
        #we need to make an ingrediant list
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        
        self.register_buffer('mask', torch.ones(
            size=(
            out_channels, 
            in_channels,
            kernel_size,
            kernel_size),
            requires_grad=False
                               ))

        self.partial_bn= nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)


    def forward(self, x, mask_input):
        masked_x = x * mask_input

        product = self.conv(masked_x)

        #find scaling product
        with torch.no_grad():
            
            mask_sum = F.conv2d(
                input=mask_input,
                weight=self.mask.to(x.device),
                stride=self.stride,
                padding=self.padding
            )

            mask_sum_no_zero = mask_sum.clone()
            mask_sum_no_zero[mask_sum == 0] = 1

            window_size = self.conv.kernel_size[0] * self.conv.kernel_size[1] * self.in_channels

            scaling_factor = window_size/mask_sum_no_zero

        out = product * scaling_factor

        mask_out = torch.zeros_like(mask_sum)
        mask_out[mask_sum >0] = 1

        out = self.partial_bn(out)
        out = self.activation(out)

        return out, mask_out


class tiny_unet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        
        # Initial convolution to go from input channels to first encoder channels
        self.initial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            padding=1)
        
        # Encoder (stays the same - these look correct)
        self.encoder_1 = partial_convolutions(
            in_channels=16, out_channels=32,
            kernel_size=5, stride=2, padding=2, negative_slope=0.2)
        
        self.encoder_2 = partial_convolutions(
            in_channels=32, out_channels=64,
            kernel_size=3, stride=2, padding=1, negative_slope=0.2)
        
        self.encoder_3 = partial_convolutions(
            in_channels=64, out_channels=128,
            kernel_size=3, stride=2, padding=1, negative_slope=0.2)
        
        self.encoder_4 = partial_convolutions(
            in_channels=128, out_channels=256,
            kernel_size=3, stride=2, padding=1, negative_slope=0.2)
        
        self.bottle_neck = partial_convolutions(
            in_channels=256, out_channels=512,
            kernel_size=3, stride=2, padding=1, negative_slope=0.2)
        
        # Decoder - FIXED CHANNEL COUNTS
        # Block 1: 512 → 256, concat x_4 (256) → 512 total
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=512,      # From bottleneck
            out_channels=256,
            kernel_size=2, stride=2)
        self.decoder_1 = decoder_double_convolution(
            in_channels=512,      # 256 + 256 from skip
            out_channels=256)
        
        # Block 2: 256 → 128, concat x_3 (128) → 256 total
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=256,      # From decoder_1
            out_channels=128,
            kernel_size=2, stride=2)
        self.decoder_2 = decoder_double_convolution(
            in_channels=256,      # 128 + 128 from skip
            out_channels=128)
        
        # Block 3: 128 → 64, concat x_2 (64) → 128 total
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=128,      # From decoder_2
            out_channels=64,
            kernel_size=2, stride=2)
        self.decoder_3 = decoder_double_convolution(
            in_channels=128,      # 64 + 64 from skip
            out_channels=64)
        
        # Block 4: 64 → 32, concat x_1 (32) → 64 total
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=64,       # From decoder_3
            out_channels=32,
            kernel_size=2, stride=2)
        self.decoder_4 = decoder_double_convolution(
            in_channels=64,       # 32 + 32 from skip
            out_channels=32)
        
        # Final output layer
        self.decoder_output = decoder_final_layer_convolution(
            in_channels=32, 
            out_channels=out_channels)  # Use constructor parameter!

    def forward(self, x, mask):
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder
        x_1, m_1 = self.encoder_1(x, mask)
        x_2, m_2 = self.encoder_2(x_1, m_1)
        x_3, m_3 = self.encoder_3(x_2, m_2)
        x_4, m_4 = self.encoder_4(x_3, m_3)
        x_btt_nck, m_btt_nck = self.bottle_neck(x_4, m_4)
        
        # Decoder
        x = self.up_transpose_1(x_btt_nck)
        x = torch.cat([x, x_4], dim=1)  # 256 + 256 = 512
        x = self.decoder_1(x)            # 512 → 256
        
        x = self.up_transpose_2(x)
        x = torch.cat([x, x_3], dim=1)  # 128 + 128 = 256
        x = self.decoder_2(x)            # 256 → 128
        
        x = self.up_transpose_3(x)
        x = torch.cat([x, x_2], dim=1)  # 64 + 64 = 128
        x = self.decoder_3(x)            # 128 → 64
        
        x = self.up_transpose_4(x)
        x = torch.cat([x, x_1], dim=1)  # 32 + 32 = 64
        x = self.decoder_4(x)            # 64 → 32
        
        x = self.decoder_output(x)       # 32 → out_channels
        
        return x

