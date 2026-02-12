# visualize_results.py
# Phase 1 Compatible: Visualizes Real/Imag Pressure from 8-Channel Output

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Check your filenames here! 
# Make sure these match the files where you saved the classes
from model_v1a import tiny_unet       # Was model_v1a
from train_v1a import GaussianDataset    # Was train_v1a

# ========== SETUP ==========

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load trained model
# CRITICAL UPDATE: Must match the Phase 1 Architecture (2 In, 8 Out)
model = tiny_unet(in_channels=2, out_channels=8).to(device)

try:
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    print("✓ Model loaded successfully")
except RuntimeError as e:
    print("\n!!! ERROR: Model Size Mismatch !!!")
    print("You likely have an old 1-channel checkpoint.")
    print("Please run train.py first to train the new 8-channel model.")
    exit()

model.eval()

# Load validation data
val_dataset = GaussianDataset('./data', mode='val')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"✓ Loaded {len(val_dataset)} validation samples")

# ========== COMPUTE METRICS ==========

print("\nComputing metrics on validation set...")
mse_list = []
psnr_list = []

with torch.no_grad():
    for batch in val_loader:
        inputs = batch['input'].to(device)
        masks = batch['mask'].to(device)
        targets = batch['target'].to(device)
        
        # 1. Full Prediction [B, 8, 32, 32]
        full_outputs = model(inputs, masks)
        
        # 2. Slice Pressure Channels [B, 2, 32, 32]
        # Channel 0 (Real Pressure) + Channel 4 (Imag Pressure)
        pred_real = full_outputs[:, 0:1, :, :]
        pred_imag = full_outputs[:, 4:5, :, :]
        outputs = torch.cat([pred_real, pred_imag], dim=1)
        
        # 3. Compute Metrics (on Pressure only)
        mse = torch.mean((outputs - targets) ** 2).item()
        mse_list.append(mse)
        
        # PSNR
        # For normalized data [0,1], MAX=1. 
        # Note: If your data goes -1 to 1, range is 2. Adjust if needed.
        # Assuming normalized roughly to 1 for now.
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        psnr_list.append(psnr)

avg_mse = np.mean(mse_list)
avg_psnr = np.mean(psnr_list)
print(f"\nValidation Metrics:")
print(f"  Average MSE:  {avg_mse:.6f}")
print(f"  Average PSNR: {avg_psnr:.2f} dB")

# ========== VISUALIZE SAMPLES ==========

print("\nGenerating visualizations...")

# Create figure with 4 samples
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i >= 4:
            break
        
        inputs = batch['input'].to(device)
        masks = batch['mask'].to(device)
        targets = batch['target'].to(device)
        
        # Predict & Slice
        full_outputs = model(inputs, masks)
        pred_real = full_outputs[:, 0:1, :, :]
        pred_imag = full_outputs[:, 4:5, :, :]
        outputs = torch.cat([pred_real, pred_imag], dim=1)
        
        # Move to CPU for plotting
        # Index [0, 0] gets Batch 0, Channel 0 (Real Part)
        # We visualize the REAL part for now
        input_np = inputs[0, 0].cpu().numpy()
        mask_np = masks[0, 0].cpu().numpy()
        target_np = targets[0, 0].cpu().numpy()
        output_np = outputs[0, 0].cpu().numpy()
        
        # Calculate metrics for this specific sample
        mse = np.mean((target_np - output_np) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        
        # Calculate absolute error
        error = np.abs(target_np - output_np)
        max_error = np.max(error)
        
        # Plot 1: Sparse Input (Real)
        axes[i, 0].imshow(input_np, cmap='hot', vmin=0, vmax=1)
        sensor_y, sensor_x = np.where(mask_np > 0)
        axes[i, 0].scatter(sensor_x, sensor_y, c='cyan', s=20, marker='x')
        axes[i, 0].set_title(f'Input (Real)\n(16 sensors)', fontsize=12)
        axes[i, 0].axis('off')
        
        # Plot 2: Reconstruction (Real)
        axes[i, 1].imshow(output_np, cmap='hot', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Reconstruction (Real)\nPSNR={psnr:.1f} dB', fontsize=12)
        axes[i, 1].axis('off')
        
        # Plot 3: Ground Truth (Real)
        axes[i, 2].imshow(target_np, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title('Ground Truth (Real)', fontsize=12)
        axes[i, 2].axis('off')
        
        # Plot 4: Absolute Error
        im = axes[i, 3].imshow(error, cmap='viridis', vmin=0, vmax=0.3)
        axes[i, 3].set_title(f'Abs Error\nMax={max_error:.3f}', fontsize=12)
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

# Add overall title
fig.suptitle(f'Validation Results (Avg PSNR: {avg_psnr:.1f} dB)', fontsize=16, y=0.995)

plt.tight_layout()
plt.savefig('reconstructions.png', dpi=150, bbox_inches='tight')
print('\n✓ Saved reconstructions.png')

# ========== PLOT ERROR HISTOGRAM ==========

fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(psnr_list, bins=30, edgecolor='black', alpha=0.7)
ax.set_xlabel('PSNR (dB)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'PSNR Distribution\nMean={avg_psnr:.2f} dB', fontsize=14)
ax.axvline(avg_psnr, color='red', linestyle='--', linewidth=2, label=f'Mean')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('psnr_distribution.png', dpi=150)
print('✓ Saved psnr_distribution.png')

print('\nVisualization Complete!')