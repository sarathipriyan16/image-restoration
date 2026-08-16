import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ======================================================================
# 1. DATASET LOADER
# ======================================================================
class KLADatasetTrial18(Dataset):
    def __init__(self, input_paths, target_paths, a_p0_1, b_p99_9):
        self.inputs = input_paths
        self.targets = target_paths
        self.a = a_p0_1
        self.b = b_p99_9
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx):
        in_arr = np.load(self.inputs[idx]).astype(np.float32)
        tg_arr = np.load(self.targets[idx]).astype(np.float32)
        if in_arr.max() > 2.0: in_arr /= 255.0
        if tg_arr.max() > 2.0: tg_arr /= 255.0
        in_norm = (in_arr - self.a) / (self.b - self.a + 1e-6)
        tg_norm = (tg_arr - self.a) / (self.b - self.a + 1e-6)
        return torch.tensor(in_norm).unsqueeze(0), torch.tensor(tg_norm).unsqueeze(0)

# ======================================================================
# 2. HIGH-END PIPELINE: CONVNEXT-V2 BLOCKS
# ======================================================================
class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

class GlobalResponseNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        
    def forward(self, x):
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return res + x

class LaplacianEdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32))
    def forward(self, x): 
        # FIXED: Cast the weight dynamically to match the input tensor's device AND precision
        return F.conv2d(x, self.weight.to(device=x.device, dtype=x.dtype), padding=1)

# ======================================================================
# 3. TRIAL 18 MODEL ARCHITECTURE
# ======================================================================
class RepPhyDAS_ConvNeXt(nn.Module):
    def __init__(self, img_channel=1, width=32):
        super().__init__()
        self.edge_extractor = LaplacianEdgeExtractor()
        self.intro = nn.Conv2d(2, width, 3, padding=1)
        
        self.enc1 = nn.Sequential(ConvNeXtV2Block(width), ConvNeXtV2Block(width))
        self.down = nn.Conv2d(width, width * 2, 2, stride=2)
        self.bottleneck = nn.Sequential(ConvNeXtV2Block(width*2), ConvNeXtV2Block(width*2))
        
        self.up = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = nn.Sequential(ConvNeXtV2Block(width), ConvNeXtV2Block(width))
        
        self.pixel_shuffle_up = nn.Sequential(
            nn.Conv2d(width, width * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(width, width, 3, padding=1)
        )
        self.out_head = nn.Conv2d(width, img_channel, 3, padding=1)

    def forward(self, y):
        edge_map = self.edge_extractor(y)
        x = self.intro(torch.cat([y, edge_map], dim=1))
        
        e1 = self.enc1(x)
        b = self.bottleneck(self.down(e1))
        d1 = self.dec1(self.up(b) + e1)
        
        out = self.pixel_shuffle_up(d1)
        return torch.clamp(self.out_head(out), 0.0, 1.0)

# ======================================================================
# 4. ADVANCED LOSS FUNCTIONS
# ======================================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplacian = LaplacianEdgeExtractor()
    def forward(self, pred, target):
        return F.l1_loss(self.laplacian(pred), self.laplacian(target))

class CombinedRestorationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.edge = EdgeLoss()
    def forward(self, pred, target):
        return self.charb(pred, target) + 0.2 * self.edge(pred, target)

# ======================================================================
# 5. TRAINING LOOP
# ======================================================================
def train_trial18():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Trial 18 (ConvNeXt-V2 + Edge Loss) on device: {device}")
    
    with open("dataset_statistics.json", "r") as f: stats = json.load(f)
    a, b = stats["a_p0_1"], stats["b_p99_9"]
    
    train_files = glob.glob("./dataset/train/**/NoisyLR/*.npy", recursive=True)
    train_files = [f for f in train_files if "__MACOSX" not in f]
    targets = [f.replace("NoisyLR", "GT") for f in train_files]
    
    dataset = KLADatasetTrial18(train_files, targets, a, b)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    
    model = RepPhyDAS_ConvNeXt(width=32).to(device)
    try:
        model = torch.compile(model, mode="default")
        print("[!] PyTorch 2.x Model Compilation Enabled.")
    except: pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) 
    
    # FIXED: Added .to(device) to the loss function to push all internal layers to the GPU
    criterion = CombinedRestorationLoss().to(device) 
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 20
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Trial 18 - Epoch {epoch+1}/{epochs}")
        for y, x_gt in pbar:
            y, x_gt = y.to(device), x_gt.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                x_hat = model(y)
                loss = criterion(x_hat, x_gt)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        epoch_loss = running_loss / len(loader)
        print(f"--> Epoch [{epoch+1}/{epochs}] Avg Loss: {epoch_loss:.6f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'model_state_dict': model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(), 
                'a': a, 'b': b
            }, "trial_18_model.pth")
            print("    [!] New best model saved as: trial_18_model.pth")

train_trial18()