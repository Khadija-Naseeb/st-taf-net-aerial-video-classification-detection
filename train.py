import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

from st_taf_net import ST_TAF_Net
from loss import JointLoss
from mod20_dataset import MOD20Dataset


# ===========================================================
#  CONFIGURATION — Edit these settings before running
# ===========================================================

# !! IMPORTANT: Change this to the folder where your MOD20 videos are stored !!
# The folder should contain subfolders named after each event class, e.g.:
#   D:/MOD20/
#       Ambulance/
#           video001.mp4
#       Fire_Truck/
#           video002.mp4
#       ...
DATA_ROOT = "D:/MOD20"   # <-- CHANGE THIS PATH

BATCH_SIZE   = 4
NUM_FRAMES   = 8          # Number of frames to sample per video clip
SPATIAL_SIZE = (256, 256) # Height x Width
EVENT_CLASSES = 20        # Number of scene/event classes (MOD20 has 20)
DET_CLASSES   = 10        # Number of detection object classes
EPOCHS        = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4         # Set to 0 if running on Windows and getting errors

# ===========================================================


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, data in enumerate(dataloader):
        x           = data[0].to(device)
        cls_target  = data[1].to(device)
        hm_target   = data[2].to(device)
        off_target  = data[3].to(device)
        size_target = data[4].to(device)
        mask        = data[5].to(device)

        optimizer.zero_grad()

        with autocast():
            preds_cls, heatmaps, offsets, sizes = model(x)
            loss, l_cls, l_det, l_hm, l_off, l_size = criterion(
                preds_cls, heatmaps, offsets, sizes,
                cls_target, hm_target, off_target, size_target, mask
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"  Epoch [{epoch:02d}] | Batch [{batch_idx}/{len(dataloader)}] | "
                  f"Loss: {loss.item():.4f} | Cls: {l_cls.item():.4f} | Det: {l_det.item():.4f}")

    return total_loss / len(dataloader)


def main():
    print("=" * 60)
    print("  ST-TAF Net — Real Dataset Training")
    print("=" * 60)

    # Check dataset path exists
    if not os.path.exists(DATA_ROOT):
        print(f"\n[ERROR] Dataset folder not found: '{DATA_ROOT}'")
        print("Please open train.py and change the DATA_ROOT variable to your dataset path.\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Load Dataset
    print("Loading MOD20 dataset...")
    train_dataset = MOD20Dataset(
        data_root=DATA_ROOT,
        split="train",
        num_frames=NUM_FRAMES,
        spatial_size=SPATIAL_SIZE,
        augment=True
    )
    val_dataset = MOD20Dataset(
        data_root=DATA_ROOT,
        split="val",
        num_frames=NUM_FRAMES,
        spatial_size=SPATIAL_SIZE,
        augment=False
    )

    if len(train_dataset) == 0:
        print("[ERROR] No videos were found. Check your DATA_ROOT folder structure.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    # Build Model
    model = ST_TAF_Net(in_channels=3, event_classes=EVENT_CLASSES, detection_classes=DET_CLASSES)
    model.to(device)
    print(f"\nModel initialized — Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = JointLoss()
    scaler    = GradScaler()

    best_loss = float("inf")

    # Training Loop
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
        scheduler.step()

        print(f"  >> Epoch {epoch} Average Training Loss: {avg_train_loss:.4f}")

        # Save best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  >> Best model saved! (loss={best_loss:.4f})")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Best model saved to: best_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
