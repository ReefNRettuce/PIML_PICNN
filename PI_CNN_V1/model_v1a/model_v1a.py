# model_v1.py
# PICNN U-Net with Partial Convolutions for Sparse Array Reconstruction

import torch 
import torch.nn as nn
import torch.nn.functional as F

# ========== DECODER BUILDING BLOCKS ==========

class decoder_double_convolution(nn.Module):
    """Standard double convolution block used in decoder"""
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
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
    
    def forward(self, x):
        return self.convolution_operation(x)


class decoder_final_layer_convolution(nn.Module):
    """Final 1x1 convolution to get output channels with sigmoid activation"""
    def __init__(self, in_channels, out_channels):
        super(decoder_final_layer_convolution, self).__init__()

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels,
                      kernel_size=1, 
                      stride=1,
                      padding=0),
            
        )
    
    def forward(self, x):
        return self.final_conv(x)


# ========== PARTIAL CONVOLUTION LAYER ==========

class partial_convolutions(nn.Module):
    """
    Partial Convolution layer with mask propagation.
    Based on Liu et al. "Image Inpainting for Irregular Holes Using Partial Convolutions"
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, negative_slope, bias=False):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        # Main convolution for features
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        
        # CRITICAL FIX: Mask weight is [1, 1, K, K] - single channel, all ones
        # This convolves the mask to count valid pixels in each window
        self.register_buffer('mask_weight', torch.ones(
            size=(1, 1, kernel_size, kernel_size),
            requires_grad=False
        ))

        self.partial_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x, mask_input):
        """
        Args:
            x: Input features [B, in_channels, H, W]
            mask_input: Binary mask [B, 1, H, W] (1=valid, 0=missing)
        
        Returns:
            out: Output features [B, out_channels, H', W']
            mask_out: Updated mask [B, 1, H', W']
        """
        # Broadcast mask to all input channels for element-wise masking
        # [B, 1, H, W] → [B, in_channels, H, W]
        mask_broadcast = mask_input.repeat(1, self.in_channels, 1, 1)
        
        # Zero out features at invalid locations
        masked_x = x * mask_broadcast

        # Apply convolution to masked features
        product = self.conv(masked_x)

        # Compute scaling factor based on valid pixels in receptive field
        with torch.no_grad():
            # Count valid pixels in each window using mask convolution
            # Input: [B, 1, H, W], Weight: [1, 1, K, K] → Output: [B, 1, H', W']
            mask_sum = F.conv2d(
                input=mask_input,
                weight=self.mask_weight.to(x.device),
                stride=self.stride,
                padding=self.padding
            )

            # Avoid division by zero
            mask_sum_no_zero = mask_sum.clone()
            mask_sum_no_zero[mask_sum == 0] = 1

            # Window size = total pixels in receptive field
            window_size = self.kernel_size * self.kernel_size

            # Scaling factor compensates for missing pixels
            # If window has 10/25 valid pixels, scale output by 25/10 = 2.5
            scaling_factor = window_size / mask_sum_no_zero

        # Scale the convolution output
        out = product * scaling_factor

        # Update mask: output pixel is valid if ANY input pixel in window was valid
        mask_out = torch.zeros_like(mask_sum)
        mask_out[mask_sum > 0] = 1

        # Batch normalization and activation
        out = self.partial_bn(out)
        out = self.activation(out)

        return out, mask_out


# ========== MAIN U-NET ARCHITECTURE ==========

class tiny_unet(nn.Module):
    """
    U-Net with Partial Convolutions for sparse array reconstruction.
    
    Architecture:
    - Input: [B, 1, 32, 32] sparse field + [B, 1, 32, 32] mask
    - Encoder: 16→32→64→128→256→512 (5 downsample steps)
    - Decoder: 512→256→128→64→32→out_channels (5 upsample steps)
    - Skip connections at each level
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Initial convolution: in_channels → 16
        self.initial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            padding=1)
        
        # ===== ENCODER (Downsampling Path) =====
        # Each partial conv: stride=2 halves spatial dimensions
        
        # Level 1: 16→32, [32,32]→[16,16]
        self.encoder_1 = partial_convolutions(
            in_channels=16, out_channels=32,
            kernel_size=5, stride=2, padding=2, negative_slope=0.2)
        
        # Level 2: 32→64, [16,16]→[8,8]
        self.encoder_2 = partial_convolutions(
            in_channels=32, out_channels=64,
            kernel_size=3, stride=2, padding=1, negative_slope=0.2)
        
        # Level 3: 64→128, [8,8]→[4,4]
        self.encoder_3 = partial_convolutions(
            in_channels=64, out_channels=128,
            kernel_size=3, stride=2, padding=1, negative_slope=0.2)
        
        # Level 4: 128→256, [4,4]→[2,2]
        self.encoder_4 = partial_convolutions(
            in_channels=128, out_channels=256,
            kernel_size=3, stride=2, padding=1, negative_slope=0.2)
        
        # Bottleneck: 256→512, [2,2]→[1,1]
        self.bottle_neck = partial_convolutions(
            in_channels=256, out_channels=512,
            kernel_size=3, stride=2, padding=1, negative_slope=0.2)
        
        # ===== DECODER (Upsampling Path) =====
        
        # Decoder Block 1: [1,1]→[2,2]
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2, stride=2)
        self.decoder_1 = decoder_double_convolution(
            in_channels=512,      # 256 (upsampled) + 256 (skip from encoder_4)
            out_channels=256)
        
        # Decoder Block 2: [2,2]→[4,4]
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2, stride=2)
        self.decoder_2 = decoder_double_convolution(
            in_channels=256,      # 128 (upsampled) + 128 (skip from encoder_3)
            out_channels=128)
        
        # Decoder Block 3: [4,4]→[8,8]
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2, stride=2)
        self.decoder_3 = decoder_double_convolution(
            in_channels=128,      # 64 (upsampled) + 64 (skip from encoder_2)
            out_channels=64)
        
        # Decoder Block 4: [8,8]→[16,16]
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2, stride=2)
        self.decoder_4 = decoder_double_convolution(
            in_channels=64,       # 32 (upsampled) + 32 (skip from encoder_1)
            out_channels=32)
        
        # Final upsampling to original resolution: [16,16]→[32,32]
        self.final_upsample = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=2, stride=2)
        
        # Final output layer
        self.decoder_output = decoder_final_layer_convolution(
            in_channels=16, 
            out_channels=out_channels)

    def forward(self, x, mask):
        """
        Args:
            x: Sparse input field [B, in_channels, 32, 32]
            mask: Binary mask [B, 1, 32, 32]
        
        Returns:
            Reconstructed field [B, out_channels, 32, 32]
        """
        # Initial convolution (no masking here, just expand channels)
        x = self.initial_conv(x)  # [B, in_ch, 32, 32] → [B, 16, 32, 32]
        
        # ===== ENCODER =====
        x_1, m_1 = self.encoder_1(x, mask)      # [B, 32, 16, 16]
        x_2, m_2 = self.encoder_2(x_1, m_1)    # [B, 64, 8, 8]
        x_3, m_3 = self.encoder_3(x_2, m_2)    # [B, 128, 4, 4]
        x_4, m_4 = self.encoder_4(x_3, m_3)    # [B, 256, 2, 2]
        x_btt_nck, m_btt_nck = self.bottle_neck(x_4, m_4)  # [B, 512, 1, 1]
        
        # ===== DECODER =====
        
        # Block 1: Upsample + Skip + DoubleConv
        x = self.up_transpose_1(x_btt_nck)     # [B, 256, 2, 2]
        x = torch.cat([x, x_4], dim=1)          # [B, 512, 2, 2]
        x = self.decoder_1(x)                   # [B, 256, 2, 2]
        
        # Block 2
        x = self.up_transpose_2(x)              # [B, 128, 4, 4]
        x = torch.cat([x, x_3], dim=1)          # [B, 256, 4, 4]
        x = self.decoder_2(x)                   # [B, 128, 4, 4]
        
        # Block 3
        x = self.up_transpose_3(x)              # [B, 64, 8, 8]
        x = torch.cat([x, x_2], dim=1)          # [B, 128, 8, 8]
        x = self.decoder_3(x)                   # [B, 64, 8, 8]
        
        # Block 4
        x = self.up_transpose_4(x)              # [B, 32, 16, 16]
        x = torch.cat([x, x_1], dim=1)          # [B, 64, 16, 16]
        x = self.decoder_4(x)                   # [B, 32, 16, 16]
        
        # Final upsample to original resolution
        x = self.final_upsample(x)              # [B, 16, 32, 32]
        
        # Final output layer
        x = self.decoder_output(x)              # [B, out_channels, 32, 32]
        
        return x


# ========== TESTING CODE ==========

if __name__ == '__main__':
    """Test the model with dummy data"""
    print("Testing tiny_unet architecture...")
    
    # Create model
    model = tiny_unet(in_channels=2, out_channels=8)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 32, 32)
    dummy_mask = torch.ones(batch_size, 1, 32, 32)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Mask shape: {dummy_mask.shape}")
    
    # Forward pass
    output = model(dummy_input, dummy_mask)
    
    print(f"Output shape: {output.shape}")
    print(f"\n✓ Model test passed!")