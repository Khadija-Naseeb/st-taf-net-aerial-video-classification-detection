# ST-TAF Net — Spatio-Temporal Transformer-based Anchor-Free Network
### Aerial Video Classification & Object Detection | MOD20 Dataset

---
## Authors

- Khadija Naseeb - 25011017
- Fatema Ebrahim - 25010868
---

## Overview
ST-TAF Net is a unified deep learning framework for aerial video analytics that combines:
- **Scene-Level Classification** (20 action/event categories)
- **Anchor-Free Object Detection** (heatmap-based, scale-invariant)

**Documented Performance on MOD20:**

| Metric | MSTN Baseline | **ST-TAF Net** | Improvement |
|---|---|---|---|
| Classification Accuracy | 78.1% | **82.4%** | +4.3% |
| Detection mAP | 29.5 | **34.2** | +4.7 |
| Inference Speed | 11 FPS | **22 FPS** | 2x faster |

---

## File Structure

```
├── st_taf_net.py        # Model with ablation flags (use_se, use_offset, task_mode)
├── loss.py              # Joint loss + uncertainty (adaptive) weighting option
├── mod20_dataset.py     # Dataset loader: full augmentations + bbox supervision
├── eval.py              # Evaluator: top-1 acc, mAP@0.5, FPS
├── train.py             # Training loop. Importable train_model(cfg)
├── run_ablation.py      # Runs all 6 rows of Table 2 in the paper
├── test_forward.py      # Architecture sanity check (no dataset needed)
├── requirements.txt     # Python dependencies
└── README.md            # This file
└── assets/
```

---

## Step 1: Install Python

If you don't have Python installed:
1. Go to: **https://www.python.org/downloads/**
2. Download **Python 3.10** or newer
3. During installation, check **"Add Python to PATH"**

Verify installation:
```powershell
python --version
```

---

## Step 2: Install Dependencies

```powershell
cd "C:\Users\My-PC\Downloads\ST"
pip install -r requirements.txt
```

This installs: `torch`, `torchvision`, `opencv-python`, `decord`, `numpy`, `matplotlib`, `scikit-learn`

> **Note on `decord`**: If decord fails to install on Windows, the code falls back to OpenCV automatically.

---

## Step 3: Download the Dataset

MOD20 is a **Multi-view Outdoor Dataset** containing 2,324 aerial video clips across 20 action categories, collected via drone cameras.

### Option A — Contact the Authors (Official Channel)
- Paper: *"Multi-view Action Recognition using Cross-view Video Prediction"*
- Search: **https://arxiv.org/search/?searchtype=all&query=MOD20+aerial+action**

### Option B — Use a Compatible Public Dataset (Recommended for Testing)

| Dataset | Link | Classes | Task |
|---|---|---|---|
| **UCF-ARG** (Aerial Action) | https://www.crcv.ucf.edu/data/UCF_ARG.php | 10 | Action Recognition |
| **ERA Dataset** | https://lcmou.github.io/ERA_Dataset/ | 25 | Aerial Event Recognition |
| **VisDrone-VID** | https://github.com/VisDrone/VisDrone-Dataset | 10 | Object Detection |

> ⭐ **ERA** is the closest publicly available substitute. For **VisDrone-VID** you get bounding-box annotations out of the box — useful if you want non-trivial detection metrics.

---

## Step 4: Organize Your Videos

Folder structure (one folder per class):

```
MOD20/
    ambulance/
        video001.mp4
        video002.mp4
    crowd/
        video010.mp4
    fire/
        video020.mp4
    annotations.json    # OPTIONAL — see below
```

The folder names automatically become the class labels.

### Optional: Bounding-Box Annotations

For real detection metrics (mAP@0.5), drop an `annotations.json` at the dataset root:

```json
{
  "ambulance/video001.mp4": [
    [0, 120, 240, 80, 60],
    [3, 510, 305, 45, 90]
  ],
  "fire/video020.mp4": [
    [5, 50, 50, 200, 200]
  ]
}
```

Each entry is `[class_id, x, y, w, h]` in **absolute pixel coordinates of the original frame**. Boxes are treated as constant across the clip's frames (a reasonable simplification for short aerial action clips). Videos missing from the file simply contribute zero detection supervision and do not break training.

If you skip this file, the classification head still trains correctly; the detection metrics will report zero/near-zero mAP because there are no targets.

---

## Step 5: Quick Architecture Test (No Dataset Needed)

```powershell
python test_forward.py
```

Expected output (~3 seconds on CPU):

```
============================================================
  Full ST-TAF Net forward + backward sanity check
============================================================
Input video tensor : (2, 3, 4, 64, 64)
Target heatmap     : (2, 10, 8, 8)
...
Forward pass completed. Output keys & shapes:
  event_cls    : (2, 20)
  heatmap      : (2, 10, 8, 8)
  offset       : (2, 2, 8, 8)
  size         : (2, 2, 8, 8)

Backward pass completed.
  Total loss        : ...

============================================================
  Ablation flag sanity check
============================================================
  full            loss=...   keys=['event_cls', 'heatmap', 'offset', 'size']
  no_se           loss=...   keys=['event_cls', 'heatmap', 'offset', 'size']
  cls_only        loss=...   keys=['event_cls']
  det_only        loss=...   keys=['heatmap', 'offset', 'size']
  no_offset       loss=...   keys=['event_cls', 'heatmap', 'offset', 'size']
  fixed_weights   loss=...   keys=['event_cls', 'heatmap', 'offset', 'size']

All configurations passed.
```

---

## Step 6: Train the Full Model

Edit the `DATA_ROOT` variable in `train.py` (top of `DEFAULT_CONFIG`), or pass `--data-root` on the CLI:

```powershell
python train.py --data-root "C:/Users/YourName/Downloads/MOD20"
```

The best model (chosen by validation metric, not training loss) is saved at `runs/sttaf_full/best_model.pth` along with `summary.json`.

### Available CLI flags

| Flag | Purpose |
|---|---|
| `--data-root PATH` | Dataset root |
| `--epochs N` | Training epochs (default 40) |
| `--batch-size N` | Batch size (default 4) |
| `--lr X` | Learning rate (default 3e-4) |
| `--no-se` | Disable the Temporal SE module |
| `--no-offset` | Disable the offset branch |
| `--task-mode {joint,cls_only,det_only}` | Pick which heads to train |
| `--fixed-weights` | Use fixed loss weights instead of learnable uncertainty weighting |
| `--run-name STR` | Subfolder under `runs/` for checkpoints |

---

## Step 7: Run the Full Ablation Study (Table 2)

```powershell
python run_ablation.py --data-root "C:/Users/YourName/Downloads/MOD20" --epochs 40
```

This trains six models in sequence:

1. **Full ST-TAF Net** — all components on, adaptive loss weighting.
2. **Without SE module** — `use_se=False`.
3. **Without joint training (cls only)** — `task_mode='cls_only'`.
4. **Without joint training (det only)** — `task_mode='det_only'`.
5. **Without offset branch** — `use_offset=False`.
6. **Fixed loss weights (λ₁=λ₂=1)** — `adaptive_weights=False`.

Each row's checkpoint and `summary.json` are saved under `runs/ablation/<run_name>/`. A combined `runs/ablation/ablation_results.json` is written, and a formatted table is printed at the end.

For a **smoke test** (1 epoch per config) before committing to the full study:

```powershell
python run_ablation.py --data-root /path/to/dataset --quick
```

---

## Notes on the New Ablation Mechanics

A subtle but important point: the "Fixed loss weights" ablation row only makes sense if the *baseline* uses something different. The codebase therefore defaults to **learnable uncertainty weighting** (Kendall, Gal & Cipolla, 2018):

$$L = \exp(-s_{cls}) \cdot L_{cls} + s_{cls} + \exp(-s_{det}) \cdot L_{det} + s_{det}$$

where `s_cls` and `s_det` are free scalar parameters. The "Fixed loss weights" row turns this off and reverts to plain `λ₁·L_cls + λ₂·L_det`. Inside `L_det`, the `λ_size = 0.1` weighting from the paper still applies in both modes — only the cls/det balance changes.

If you want to keep the paper's loss formula literally as written, pass `--fixed-weights` for the main run *and* drop row 6 from the ablation (since it would then be identical to row 1).

---

## Hardware Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **RAM** | 8 GB | 16 GB |
| **GPU** | None (CPU works) | NVIDIA RTX with 8+ GB VRAM |
| **Storage** | 5 GB (code) | 50+ GB (full dataset) |
| **Python** | 3.8+ | 3.10 |
| **CUDA** | Not required | 11.8+ for GPU acceleration |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: torch` | Run `pip install torch torchvision` |
| `decord` install fails | `pip install decord` or ignore (OpenCV fallback activates) |
| `No videos found` error | Check `--data-root` path and folder structure (see Step 4) |
| `CUDA out of memory` | Reduce `--batch-size` from 4 to 2 |
| `num_workers` error on Windows | Set `num_workers=0` in `DEFAULT_CONFIG` |
| mAP is always 0.0 | You probably haven't supplied an `annotations.json` (see Step 4) |
| Training is very slow | No GPU detected — use Google Colab (free GPU) |

---

## Running on Google Colab

```python
!pip install torch torchvision decord opencv-python
!python run_ablation.py --data-root /content/drive/MyDrive/MOD20 --epochs 40
```

---

## References

1. Yang et al. (2025). *Aerial Video Classification by Integrating Global-Local Semantics in ConvNets.*
2. FuTH-Net: *Fusing Temporal Relations and Holistic Features for Aerial Video Classification.*
3. Zhou et al. (2019). *Objects as Points* (CenterNet) — anchor-free detection foundation.
4. Liu et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.*
5. Kendall, Gal & Cipolla (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses* — adaptive loss weighting used by the joint loss when `adaptive_weights=True`.

