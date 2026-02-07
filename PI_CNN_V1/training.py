# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from model_v1 import tiny_unet

# ========== DEVICE SETUP ==========
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found.")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU.")

# ========== DATASET ==========
class GaussianDataset(Dataset):
    def __init__(self, data_dir='./data', mode='train'):
        self.data_dir = Path(data_dir)
        self.file_list = sorted(list(self.data_dir.glob('gaussian_*.npz')))
        
        # Split 80/20 train/val
        split_idx = int(0.8 * len(self.file_list))
        if mode == 'train':
            self.file_list = self.file_list[:split_idx]
        else:
            self.file_list = self.file_list[split_idx:]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load .npz file
        data = np.load(self.file_list[idx])
        
        measurements = data['measurements']  # [16]
        mask = data['mask']                  # [32, 32]
        ground_truth = data['ground_truth']  # [32, 32]
        
        # Create sparse input field
        input_field = np.zeros((32, 32))
        input_field.flat[mask.flatten() == 1] = measurements
        
        # Convert to tensors [1, 32, 32] - add channel dimension
        input_field = torch.from_numpy(input_field).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        ground_truth = torch.from_numpy(ground_truth).float().unsqueeze(0)
        
        return {
            'input': input_field,
            'mask': mask,
            'target': ground_truth
        }

# ========== TRAINING FUNCTION ==========
def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        inputs = batch['input'].to(device)
        masks = batch['mask'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        outputs = model(inputs, masks)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print every 10 batches
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ========== VALIDATION FUNCTION ==========
def validate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            masks = batch['mask'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs, masks)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ========== PLOTTING FUNCTION ==========
def plot_losses(train_losses, val_losses, save_path='loss_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'\n✓ Loss curve saved to {save_path}')
    plt.close()

# ========== MAIN TRAINING SCRIPT ==========
def main():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 100
    PATIENCE = 15  # Stop if no improvement for 15 epochs
    
    # Create datasets
    train_dataset = GaussianDataset('./data', mode='train')
    val_dataset = GaussianDataset('./data', mode='val')
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    model = tiny_unet(in_channels=1, out_channels=1).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel has {total_params:,} parameters')
    
    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print('\n' + '='*60)
    print('Starting Training')
    print('='*60)
    
    # Training loop
    for epoch in range(MAX_EPOCHS):
        print(f'\nEpoch {epoch+1}/{MAX_EPOCHS}')
        print('-' * 50)
        
        # Train
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        print(f'Train Loss: {train_loss:.6f}')
        
        # Validate
        val_loss = validate(val_loader, model, loss_fn, device)
        val_losses.append(val_loss)
        print(f'Val Loss: {val_loss:.6f}')
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'✓ Saved best model (val_loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            print(f'No improvement ({patience_counter}/{PATIENCE})')
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f'\n{"="*60}')
            print(f'Early stopping triggered at epoch {epoch+1}')
            print(f'Best validation loss: {best_val_loss:.6f}')
            print(f'{"="*60}')
            break
    
    # Final summary
    print(f'\n{"="*60}')
    print('Training Complete!')
    print(f'{"="*60}')
    print(f'Total epochs trained: {len(train_losses)}')
    print(f'Best validation loss: {best_val_loss:.6f}')
    print(f'Final train loss: {train_losses[-1]:.6f}')
    print(f'Final val loss: {val_losses[-1]:.6f}')
    
    # Plot losses
    plot_losses(train_losses, val_losses)

if __name__ == '__main__':
    main()
