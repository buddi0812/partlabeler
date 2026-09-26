"""DINOv3 patch features (timm weights, pinned in engine.models) for matching parts across images."""
import numpy as np
import timm
from PIL import Image
import torch
import torch.nn.functional as F

from engine import hw
from engine.models import DINOV3, fetch


class Embedder:
    patch = 16

    def __init__(self, size: str = "small", device: str | None = None):
        device = device or hw.device()
        spec = DINOV3[size]
        path = fetch(spec)
        self.model = timm.create_model(spec["repo_id"].split("/", 1)[1], pretrained=True, num_classes=0,
                                       dynamic_img_size=True,
                                       pretrained_cfg_overlay={"file": str(path / "model.safetensors")})
        self.model = self.model.to(device).eval()
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=device).view(3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(3, 1, 1)
        self.device = device

    @torch.inference_mode()
    def patches(self, image, width: int = 1024):
        """L2-normalised patch features (rows, cols, C) for `image` resized to about `width` px wide,
        plus the (x, y) scale from image pixels to the resized image."""
        W, H = image.size
        nw = max(self.patch, round(width / self.patch) * self.patch)
        nh = max(self.patch, round(H * nw / W / self.patch) * self.patch)
        x = torch.from_numpy(np.array(image.convert("RGB").resize((nw, nh)))).to(self.device)
        x = ((x.permute(2, 0, 1).float() / 255 - self.mean) / self.std)[None]
        with torch.autocast("cuda", dtype=hw.dtype(self.device), enabled=self.device == "cuda"):
            tokens = self.model.forward_features(x)[0, self.model.num_prefix_tokens:]
        feats = F.normalize(tokens.float(), dim=-1).view(nh // self.patch, nw // self.patch, -1)
        return feats, (nw / W, nh / H)

    @torch.inference_mode()
    def vectors(self, images, size: int = 224, batch: int = 32, pool: str = "avg") -> np.ndarray:
        """One L2-normalised feature per image (whole images or part crops), for grouping and classifying:
        the mean of the patch tokens ("avg", best in S10), the CLS token ("cls") or both side by side ("both")."""
        out = []
        for i in range(0, len(images), batch):
            x = torch.stack([torch.from_numpy(np.array(im.convert("RGB").resize((size, size), Image.BILINEAR)))
                             for im in images[i:i + batch]]).to(self.device)
            x = (x.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
            with torch.autocast("cuda", dtype=hw.dtype(self.device), enabled=self.device == "cuda"):
                t = self.model.forward_features(x).float()
            cls, avg = F.normalize(t[:, 0], dim=-1), F.normalize(t[:, self.model.num_prefix_tokens:].mean(1), dim=-1)
            v = {"cls": cls, "avg": avg, "both": torch.cat([cls, avg], -1)}[pool]
            out.append(F.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.num_features), np.float32)
