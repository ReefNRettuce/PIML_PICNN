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
            nn.LeakyReLU(negative_slope=0.2,inplace=False),
            nn.BatchNorm2d(out_channels)
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
    def __init__(self, x):
        super().__init__()
        


    def forward(self,x):
        pass

def training():
    pass

# put dataloader here

# put training loop 

# def PDE here 

# def something something here I don't know what I'm doing. 