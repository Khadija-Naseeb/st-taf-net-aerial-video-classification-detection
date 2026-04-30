import torch
from st_taf_net import ST_TAF_Net
from loss import JointLoss
import time

def test_pipeline():
    print("Initializing ST-TAF Net...")
    batch_size = 2
    in_channels = 3
    event_classes = 20
    det_classes = 10
    
    device = torch.device("cpu")
    model = ST_TAF_Net(in_channels=in_channels, event_classes=event_classes, detection_classes=det_classes)
    criterion = JointLoss()
    
    # We will use extremely small downscaled inputs to ensure it evaluates in under 3 seconds on CPU
    # Typical shapes used in testing environments
    t = 4
    h_in, w_in = 64, 64
    h_out, w_out = h_in // 8, w_in // 8
    
    # Dummy outputs
    x_dummy = torch.randn(batch_size, in_channels, t, h_in, w_in)
    cls_dummy = torch.randint(0, event_classes, (batch_size,))
    hm_dummy = torch.zeros(batch_size, det_classes, h_out, w_out)
    off_dummy = torch.zeros(batch_size, 2, h_out, w_out)
    size_dummy = torch.ones(batch_size, 2, h_out, w_out) * 5
    mask_dummy = torch.zeros(batch_size, h_out, w_out)
    
    # Dummy Object at center (4, 4) if h_out is 8
    hm_dummy[:, 0, h_out//2, w_out//2] = 1.0 
    mask_dummy[:, h_out//2, w_out//2] = 1.0  
    
    print("\n--- Dummy Data Shapes ---")
    print(f"Input Video Tensor: {x_dummy.shape}")
    print(f"Target Heatmap: {hm_dummy.shape}")
    print(f"Target Offset: {off_dummy.shape}")
    print(f"Target Size: {size_dummy.shape}")
    print("-------------------------\n")
    
    print("Performing Forward Pass...")
    start_time = time.time()
    
    model.train()
    preds_cls, heatmaps, offsets, sizes = model(x_dummy)
    
    print("Forward Pass Completed! Output shapes:")
    print(f"Event Cls Prediction: {preds_cls.shape}")
    print(f"Heatmap Prediction: {heatmaps.shape}")
    print(f"Offset Prediction: {offsets.shape}")
    print(f"Size Prediction: {sizes.shape}\n")
    
    print("Calculating Multi-Task Loss...")
    loss, l_cls, l_det, l_hm, l_off, l_size = criterion(
        preds_cls, heatmaps, offsets, sizes,
        cls_dummy, hm_dummy, off_dummy, size_dummy, mask_dummy
    )
    
    # Do backward pass
    loss.backward()
    
    print(f"Backward Pass Completed! Parameters processed.")
    print(f"Total Combined Loss: {loss.item():.4f}")
    print(f" - Cls Loss (Cross Entropy): {l_cls.item():.4f}")
    print(f" - Det Loss: {l_det.item():.4f}")
    print(f"    - Heatmap (Focal) Loss: {l_hm.item():.4f}")
    print(f"    - Offset (L1) Loss: {l_off.item():.4f}")
    print(f"    - Size (L1) Loss: {l_size.item():.4f}")
    
    end_time = time.time()
    print(f"\nTime taken for evaluation: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    test_pipeline()
