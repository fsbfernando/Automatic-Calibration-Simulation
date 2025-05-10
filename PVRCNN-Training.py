import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

# ==== CONFIGURAÇÕES ====
DATASET_DIR = Path.home() / "Fernando-Braga" / "Automatic Calibration" / "Dataset" / "ds0_aug"
POINTCLOUD_DIR = DATASET_DIR / "pointcloud"
ANNOTATION_DIR = DATASET_DIR / "ann"
OUTPUT_DIR = Path("output/robotbase_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MIN_POINTS = 2048

# ==== DEFINIÇÃO DE DATASET ====
class RobotBaseDataset(Dataset):
    def __init__(self, pcd_dir, ann_dir):
        self.pcd_dir = Path(pcd_dir)
        self.ann_dir = Path(ann_dir)
        self.samples = sorted([f.name.replace(".pcd", "") for f in self.pcd_dir.glob("*.pcd") if (ann_dir / f"{f.name}.json").exists()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_name = self.samples[idx]
        pcd_path = self.pcd_dir / f"{base_name}.pcd"
        ann_path = self.ann_dir / f"{base_name}.pcd.json"

        # Leitura da nuvem de pontos
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points).astype(np.float32)

        # Corrige para ter exatamente N pontos
        N = 2048
        if points.shape[0] >= N:
            idx = np.random.choice(points.shape[0], N, replace=False)
        else:
            idx = np.random.choice(points.shape[0], N, replace=True)
        points = points[idx]

        if len(points) < MIN_POINTS:
            print(f"[WARN] {base_name}.pcd possui apenas {len(points)} pontos!")

        # Leitura da anotação
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        geom = ann['figures'][0]['geometry']
        center = np.array([geom['position']['x'], geom['position']['y'], geom['position']['z']], dtype=np.float32)
        size = np.array([geom['dimensions']['x'], geom['dimensions']['y'], geom['dimensions']['z']], dtype=np.float32)
        yaw = float(geom['rotation']['z'])

        label = np.concatenate([center, size, [yaw]]).astype(np.float32)  # shape (7,)

        return torch.from_numpy(points).float(), torch.from_numpy(label).float()

# ==== REDE SIMPLIFICADA (CORRIGIDA) ====
class DummyPVRCNN(nn.Module):
    def __init__(self):
        super(DummyPVRCNN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # [cx, cy, cz, dx, dy, dz, yaw]
        )

    def forward(self, x):
        # x: [B, N, 3]
        x = x.permute(0, 2, 1)  # → [B, 3, N]
        x = self.backbone(x).squeeze(-1)  # → [B, 128]
        return self.head(x)

# ==== CARREGAMENTO DOS DADOS ====
full_dataset = RobotBaseDataset(POINTCLOUD_DIR, ANNOTATION_DIR)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ==== TREINAMENTO ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DummyPVRCNN().to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for pts, lbl in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
        pts, lbl = pts.to(device), lbl.to(device)
        output = model(pts)
        loss = criterion(output, lbl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * pts.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for pts, lbl in val_loader:
            pts, lbl = pts.to(device), lbl.to(device)
            output = model(pts)
            loss = criterion(output, lbl)
            val_loss += loss.item() * pts.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"[{epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), OUTPUT_DIR / f"pvrcnn_epoch{epoch+1:03d}.pth")


