import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

try:
    import decord
    from decord import VideoReader, cpu
except ImportError:
    decord = None
    print("Warning: 'decord' not found. Install it: pip install decord")


class MOD20Dataset(Dataset):
    """
    MOD20 Aerial Video Dataset Loader.

    Expected folder structure:
        data_root/
            class_name_1/
                video001.mp4
                video002.mp4
            class_name_2/
                video003.mp4
            ...

    Each subfolder name is treated as an event class label automatically.
    """

    def __init__(self, data_root, split='train', num_frames=8,
                 spatial_size=(256, 256), transform=None, augment=True):
        self.data_root   = data_root
        self.split       = split
        self.num_frames  = num_frames
        self.spatial_size = spatial_size
        self.transform   = transform
        self.augment     = augment

        self.video_paths     = []
        self.event_labels    = []
        self.bbox_annotations = []  # [class_id, x, y, w, h] format

        self._load_annotations()

    def _load_annotations(self):
        """
        Scans data_root for subfolders. Each subfolder = one event class.
        Reads all .mp4/.avi/.mov/.mkv files inside each subfolder.
        """
        if not os.path.exists(self.data_root):
            print(f"[Warning] data_root not found: '{self.data_root}'")
            return

        # Sort classes alphabetically for deterministic class IDs
        classes = sorted([
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
        ])

        if len(classes) == 0:
            print(f"[Warning] No class subfolders found in '{self.data_root}'")
            return

        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        print(f"Found {len(classes)} classes: {classes}")

        for cls_name in classes:
            cls_dir = os.path.join(self.data_root, cls_name)
            cls_id  = class_to_idx[cls_name]

            for vid_name in sorted(os.listdir(cls_dir)):
                if vid_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    full_path = os.path.join(cls_dir, vid_name)
                    self.video_paths.append(full_path)
                    self.event_labels.append(cls_id)
                    # No bounding box annotations provided by default.
                    # If you have a .json annotation file, load it here and append the boxes.
                    self.bbox_annotations.append([])

        print(f"[{self.split}] Loaded {len(self.video_paths)} videos "
              f"across {len(classes)} classes.\n")

    def _apply_augmentations(self, frames):
        """Spatial augmentations: vibration simulation + resize."""
        if self.augment and np.random.rand() > 0.5:
            frames = [cv2.GaussianBlur(f, (5, 5), 0) for f in frames]

        frames = [cv2.resize(f, (self.spatial_size[1], self.spatial_size[0]))
                  for f in frames]
        return frames

    def _build_spatial_targets(self, bboxes, downsample_ratio=8):
        """Build anchor-free targets: heatmap, offset, size maps."""
        feat_h = self.spatial_size[0] // downsample_ratio
        feat_w = self.spatial_size[1] // downsample_ratio

        heatmap = np.zeros((10, feat_h, feat_w), dtype=np.float32)
        offset  = np.zeros((2, feat_h, feat_w), dtype=np.float32)
        size    = np.zeros((2, feat_h, feat_w), dtype=np.float32)
        mask    = np.zeros((feat_h, feat_w),    dtype=np.float32)

        for box in bboxes:
            cls_id, x, y, w, h = box
            cx = (x + w / 2) / downsample_ratio
            cy = (y + h / 2) / downsample_ratio
            cx_int, cy_int = int(cx), int(cy)

            if 0 <= cx_int < feat_w and 0 <= cy_int < feat_h:
                heatmap[int(cls_id), cy_int, cx_int] = 1.0
                offset[0, cy_int, cx_int] = cx - cx_int
                offset[1, cy_int, cx_int] = cy - cy_int
                size[0, cy_int, cx_int]   = w / downsample_ratio
                size[1, cy_int, cx_int]   = h / downsample_ratio
                mask[cy_int, cx_int]      = 1.0

        return heatmap, offset, size, mask

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        vid_path  = self.video_paths[idx]
        event_cls = self.event_labels[idx]
        bboxes    = self.bbox_annotations[idx]

        # Load video frames
        frames = self._load_frames(vid_path)

        # Augment and resize
        frames = self._apply_augmentations(frames)

        # Convert to [C, T, H, W] tensor
        frames_np     = np.array(frames).transpose(3, 0, 1, 2).astype(np.float32)
        frames_tensor = torch.from_numpy(frames_np) / 255.0

        # Build detection targets
        hm, off, sz, mask = self._build_spatial_targets(bboxes)

        return (
            frames_tensor,
            torch.tensor(event_cls, dtype=torch.long),
            torch.from_numpy(hm),
            torch.from_numpy(off),
            torch.from_numpy(sz),
            torch.from_numpy(mask)
        )

    def _load_frames(self, vid_path):
        """Load frames using decord (fast) or fallback to OpenCV."""
        if decord is not None:
            try:
                vr = VideoReader(vid_path, ctx=cpu(0))
                total = len(vr)
                if total >= self.num_frames:
                    indices = np.sort(np.random.choice(total, self.num_frames, replace=False))
                else:
                    indices = np.arange(total)
                frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
                return list(frames)
            except Exception as e:
                print(f"[Warning] decord failed for {vid_path}: {e}. Falling back to OpenCV.")

        # OpenCV fallback
        cap = cv2.VideoCapture(vid_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total >= self.num_frames:
            indices = sorted(np.random.choice(total, self.num_frames, replace=False))
        else:
            indices = list(range(total))

        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # Pad if too short
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else
                          np.zeros((self.spatial_size[0], self.spatial_size[1], 3), dtype=np.uint8))
        return frames
