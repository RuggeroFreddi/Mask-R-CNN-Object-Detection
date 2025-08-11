#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

from tqdm import tqdm


# =========================
# ===== Dataset COCO ======
# =========================

class CocoDataset(Dataset):
    """Dataset COCO minimale per Mask R-CNN (poligoni → maschere)."""

    def __init__(self, img_dir: str | Path, ann_path: str | Path, transforms=None):
        self.img_dir = Path(img_dir)
        self.transforms = transforms

        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        # Mappa id_categoria -> nome
        self.categories = {c["id"]: c["name"] for c in coco["categories"]}

        # Mappa image_id -> [ann, ...]
        self.ann_map: Dict[int, List[dict]] = {}
        for ann in self.annotations:
            self.ann_map.setdefault(ann["image_id"], []).append(ann)

        # Mappa category_id COCO -> label contigua [1..K] (0 è background)
        cat_ids_sorted = sorted(self.categories.keys())
        self.catid_to_contig = {cid: (i + 1) for i, cid in enumerate(cat_ids_sorted)}
        self.num_classes = 1 + len(self.catid_to_contig)

        # Indici immagini con almeno una bbox valida
        self.valid_indices: List[int] = []
        for i, img_info in enumerate(self.images):
            anns = self.ann_map.get(img_info["id"], [])
            if any(self._ann_has_valid_bbox(a) for a in anns):
                self.valid_indices.append(i)

    @staticmethod
    def _ann_has_valid_bbox(ann: dict) -> bool:
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        return (w is not None) and (h is not None) and (w > 0) and (h > 0)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self.valid_indices[idx]
        img_info = self.images[real_idx]

        img_path = self.img_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        anns = self.ann_map.get(img_info["id"], [])

        masks: List[np.ndarray] = []
        boxes: List[List[float]] = []
        labels: List[int] = []

        for ann in anns:
            if not self._ann_has_valid_bbox(ann):
                continue

            seg = ann.get("segmentation")
            if not seg or not isinstance(seg, list) or not seg[0]:
                continue

            flat = seg[0]
            poly = np.array(flat, dtype=np.float32).reshape(-1, 2)

            # Maschera dalla polygon
            mask_img = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask_img).polygon([tuple(p) for p in poly], outline=1, fill=1)
            mask = np.array(mask_img, dtype=np.uint8)
            if mask.max() == 0:
                continue

            masks.append(mask)

            # COCO bbox [x, y, w, h] -> [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            # category_id -> label contigua [1..K]
            cid = int(ann["category_id"])
            labels.append(self.catid_to_contig.get(cid, 1))

        if not boxes:
            # Se nessuna annotazione valida, passa all'elemento successivo
            return self.__getitem__((idx + 1) % len(self))

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)
        masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.as_tensor(img_info["id"], dtype=torch.int64),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


# =========================
# ===== Transforms ========
# =========================

def get_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])


# =========================
# ===== Model helpers =====
# =========================

def get_model_instance_segmentation(num_classes: int):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Classificatore box
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    # Predittore maschere
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


@torch.no_grad()
def _to_device_batch(batch_imgs, batch_targets, device):
    imgs = [im.to(device) for im in batch_imgs]
    tgts = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]
    return imgs, tgts


def collate_fn(batch):
    """Top-level per essere picklable con num_workers>0 su macOS."""
    return tuple(zip(*batch))


# =========================
# ======= Training ========
# =========================

def train_one_epoch(model, dataloader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    pbar = tqdm(dataloader, desc="⏳ Training", ncols=90)
    for imgs, targets in pbar:
        imgs, targets = _to_device_batch(imgs, targets, device)

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg = total_loss / max(steps, 1)
    print(f"📉 Loss media epoca: {avg:.4f}")
    return avg


# =========================
# ======= Valutazione =====
# =========================

@torch.no_grad()
def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """IoU tra due maschere binarie [H,W]."""
    m1b = m1.astype(bool)
    m2b = m2.astype(bool)
    inter = np.logical_and(m1b, m2b).sum()
    union = np.logical_or(m1b, m2b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


@torch.no_grad()
def evaluate_one_epoch(
    model,
    dataloader: DataLoader,
    device,
    iou_thr: float = 0.5,
    score_thr: float = 0.0
) -> Tuple[float, float, float, float]:
    """
    Valutazione semplice:
      - TP/FP/FN con matching greedy su IoU maschere (stessa classe)
      - Precision, Recall, F1 e mIoU dei match

    Nota: batch_size=1 consigliato per semplicità.
    """
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    ious_matched: List[float] = []

    for imgs, targets in tqdm(dataloader, desc="🔎 Val", ncols=90):
        imgs, targets = _to_device_batch(imgs, targets, device)
        outputs = model(imgs)  # batch size 1

        # GT
        gt_masks = targets[0]["masks"].cpu().numpy().astype(np.uint8)
        gt_labels = targets[0]["labels"].cpu().numpy()

        # Pred
        out = outputs[0]
        scores = out["scores"].detach().cpu().numpy()
        keep = scores >= score_thr
        pred_masks = out["masks"].detach().cpu().numpy()[keep, 0]  # [N,1,H,W] -> [N,H,W]
        pred_labels = out["labels"].detach().cpu().numpy()[keep]

        # Greedy matching per classe
        used_gt = np.zeros(len(gt_masks), dtype=bool)

        # Ordina pred per score desc
        order = np.argsort(scores[keep])[::-1]
        pred_masks = pred_masks[order]
        pred_labels = pred_labels[order]

        for pmask, plab in zip(pred_masks, pred_labels):
            best_iou = 0.0
            best_j = -1
            for j, (gmask, glab) in enumerate(zip(gt_masks, gt_labels)):
                if used_gt[j] or glab != plab:
                    continue
                iou = mask_iou(pmask > 0.5, gmask > 0.5)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_thr and best_j >= 0:
                total_tp += 1
                used_gt[best_j] = True
                ious_matched.append(best_iou)
            else:
                total_fp += 1

        total_fn += int((~used_gt).sum())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    miou = float(np.mean(ious_matched)) if ious_matched else 0.0

    # "Accuracy" come recall@0.5 IoU (TP/GT), più leggibile nel log
    accuracy = recall

    print(
        f"✅ Val @IoU≥{iou_thr:.2f}: "
        f"accuracy={accuracy:.3f}  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  mIoU={miou:.3f}  "
        f"[TP={total_tp} FP={total_fp} FN={total_fn}]"
    )
    return accuracy, precision, recall, f1


# =========================
# ========= Main ==========
# =========================

def main():
    # Path del dataset aumentato
    img_dir = "augmented"
    ann_path = "augmented/augmented_coco.json"

    # Dataset completo
    full_ds = CocoDataset(img_dir=img_dir, ann_path=ann_path, transforms=get_transform())
    print(f"📦 Immagini con annotazioni valide (totale): {len(full_ds)}")
    if len(full_ds) == 0:
        raise RuntimeError("Nessuna immagine valida trovata. Controlla augmented_coco.json e i path.")

    num_classes = full_ds.num_classes  # include background
    print(f"🔢 num_classes (incl. background): {num_classes}")

    # Split train/val riproducibile (90/10)
    rng = np.random.default_rng(42)
    indices = np.arange(len(full_ds))
    rng.shuffle(indices)
    split = max(1, int(0.1 * len(indices)))
    val_indices = indices[:split].tolist()
    train_indices = indices[split:].tolist()

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,          # semplifica matching
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    model = get_model_instance_segmentation(num_classes).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        train_one_epoch(model, train_loader, optimizer, device)
        # Valutazione ad ogni epoca
        evaluate_one_epoch(model, val_loader, device, iou_thr=0.5, score_thr=0.0)

    # Salvataggio in 'modelli/'
    Path("modelli").mkdir(exist_ok=True)
    out_path = Path("modelli") / "maskrcnn_mycell_1.pth"
    torch.save(model.state_dict(), out_path)
    print(f"💾 Modello salvato: {out_path.resolve()}")


if __name__ == "__main__":
    main()
