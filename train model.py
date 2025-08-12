import os
import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm


# =========================
# ====== CONFIG/CONST =====
# =========================
IMG_DIR = "augmented"                 # cartella immagini
ANN_PATH = "augmented/augmented_coco.json"  # COCO aggregato

BATCH_SIZE = 1                        # lascia 1 per debug stabile
NUM_EPOCHS = 4
LEARNING_RATE = 1e-3                  # prudente (prima 5e-3)
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 5.0                  # gradient clipping

# filtri anti-degeneri lato Dataset
MIN_BOX_SIDE = 1.0                    # scarta box con lato <= 1 px
MIN_MASK_PIXELS = 16                  # scarta mask con < N pixel ON


# =========================
# ======== DATASET ========
# =========================
class CocoDataset(Dataset):
    """
    Dataset COCO minimal per Mask R-CNN.
    - Usa segmentation poligonali per costruire le mask
    - Forza labels = 1 (una sola classe: 'organoid')
    - Filtra istanze con mask vuota o bbox piatta
    """
    def __init__(self, img_dir, ann_path, transforms=None):
        self.img_dir = Path(img_dir)
        self.transforms = transforms

        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco.get("images", [])
        self.annotations = coco.get("annotations", [])

        # indicizza annotations per image_id
        self.ann_map = {}
        for ann in self.annotations:
            self.ann_map.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def _poly_to_mask_and_box(self, poly, width, height):
        """Crea mask (H,W) uint8 e bbox [x1,y1,x2,y2] da un poligono flat."""
        if not poly or len(poly) < 6:
            return None, None

        # mask
        mask_img = Image.new("L", (width, height), 0)
        pts = list(map(tuple, np.array(poly, dtype=np.float32).reshape(-1, 2)))
        ImageDraw.Draw(mask_img).polygon(pts, outline=1, fill=1)
        mask = np.array(mask_img, dtype=np.uint8)  # (H,W)

        # bbox dai punti (robusto)
        xs = poly[0::2]
        ys = poly[1::2]
        x_min, y_min = float(min(xs)), float(min(ys))
        x_max, y_max = float(max(xs)), float(max(ys))
        # clamp all'interno immagine
        x_min = max(0.0, min(x_min, width - 1))
        y_min = max(0.0, min(y_min, height - 1))
        x_max = max(0.0, min(x_max, width - 1))
        y_max = max(0.0, min(y_max, height - 1))
        if x_max <= x_min or y_max <= y_min:
            return None, None

        return mask, [x_min, y_min, x_max, y_max]

    def __getitem__(self, idx):
        # protezione circolare in caso di sample vuoti
        loop_guard = 0
        while loop_guard < len(self):
            img_info = self.images[idx]
            file_name = img_info["file_name"]
            width, height = int(img_info["width"]), int(img_info["height"])

            img_path = self.img_dir / file_name
            img = Image.open(img_path).convert("RGB")

            anns = self.ann_map.get(img_info["id"], [])
            masks, boxes = [], []

            for ann in anns:
                seg = ann.get("segmentation")
                if not seg or not isinstance(seg, list) or len(seg) == 0:
                    continue
                poly = seg[0]
                mask, box_xyxy = self._poly_to_mask_and_box(poly, width, height)
                if mask is None:
                    continue

                # filtri anti-degeneri
                if mask.sum() < MIN_MASK_PIXELS:
                    continue
                x1, y1, x2, y2 = box_xyxy
                if (x2 - x1) <= MIN_BOX_SIDE or (y2 - y1) <= MIN_BOX_SIDE:
                    continue

                masks.append(mask)
                boxes.append([x1, y1, x2, y2])

            if len(boxes) == 0:
                # passa al sample successivo
                idx = (idx + 1) % len(self)
                loop_guard += 1
                continue

            # tensori target
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)   # [N,H,W]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)                   # [N,4]
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)             # una sola classe: 1

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([img_info["id"]], dtype=torch.int64),
                "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
                # metadati utili per debug (NON tensor, non verrà .to(device))
                "file_name_str": file_name,
            }

            if self.transforms:
                img = self.transforms(img)

            return img, target

        # Se proprio non trova niente (tutto degenero), solleva
        raise RuntimeError("Tutti i sample risultano senza istanze valide dopo i filtri.")

# =========================
# ====== TRANSFORMS =======
# =========================
def get_transform():
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


# =========================
# ======== MODELLO ========
# =========================
def get_model_instance_segmentation(num_classes=2):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    # classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    # mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
        # num_classes = 2  (background + organoid)
    )
    return model


# =========================
# ======= TRAIN LOOP ======
# =========================
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, steps = 0.0, 0

    pbar = tqdm(dataloader, desc="⏳ Training", ncols=80)
    for step, (imgs, targets) in enumerate(pbar):
        # sposta su device SOLO i tensori
        imgs = [img.to(device) for img in imgs]
        tgts = []
        for t in targets:
            tt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
            tgts.append(tt)

        # forward -> dict di loss
        loss_dict = model(imgs, tgts)
        losses = sum(loss for loss in loss_dict.values())

        # guardia: salta batch non finiti
        if not torch.isfinite(losses):
            loss_cpu = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
            print(f"\n⚠️  batch {step}: loss non finita -> {loss_cpu}")
            # info di debug sul primo target
            t0 = tgts[0]
            print("   file:", t0.get("file_name_str", "?"),
                  "| boxes:", tuple(t0["boxes"].shape),
                  "| masks:", tuple(t0["masks"].shape),
                  "| labels uniq:", t0["labels"].unique().tolist())
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += float(losses.detach().cpu())
        steps += 1
        pbar.set_postfix(loss=f"{float(losses.detach().cpu()):.4f}")

    avg = total_loss / max(1, steps)
    print(f"📉 Loss media epoca: {avg:.4f}")
    return avg


# =========================
# ========= MAIN ==========
# =========================
def main():
    # dataset + loader
    dataset = CocoDataset(IMG_DIR, ANN_PATH, transforms=get_transform())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,                      # tienilo a 0 su macOS per debug
        collate_fn=lambda x: tuple(zip(*x))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Device:", device)

    # modello
    model = get_model_instance_segmentation(num_classes=2)
    model.to(device)

    # ottimizzatore un po' più prudente
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY
    )

    # training
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch} ===")
        train_one_epoch(model, loader, optimizer, device)

    # salva pesi
    out_path = "modelli/maskrcnn_organoid.pth"
    torch.save(model.state_dict(), out_path)
    print(f"✅ Modello salvato: {out_path}")


if __name__ == "__main__":
    main()
