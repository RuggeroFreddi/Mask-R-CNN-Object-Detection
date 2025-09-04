#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inferenza Mask R-CNN su tutte le immagini in 'unlabeled img/' con maschere
ben visibili: fill opaco, contorno doppio (halo nero + bordo colorato),
stampa area per ogni istanza e box rettangolare sopra l'oggetto con la sua area.
Una sola finestra per immagine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL.ExifTags import TAGS
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# =========================
# ========= Config ========
# =========================

IMAGE_DIR = Path("test img")
MODEL_PATH = Path("modelli") / "maskrcnn_organoid.pth"
COCO_PATH = Path("augmented") / "augmented_coco.json"  # opzionale per nomi classe

NUM_CLASSES = 2          # background + 1 classe (aggiorna se necessario)
SCORE_THRESHOLD = 0.5    # confidenza minima per mostrare/contare

# Aspetto overlay
FILL_ALPHA = 0.65                # opacità del riempimento maschera
OUTLINE_WIDTH = 3.0              # spessore bordo colorato
OUTLINE_HALO_WIDTH = 6.0         # spessore halo nero sotto al bordo
OUTLINE_HALO_COLOR = "black"     # colore halo per contrasto


# =========================
# ===== Retrieve Metadata ======
# =========================

def get_pixel_info(image_file_path):
    """Estrae dimensione e area pixel dai metadati EXIF."""
    try:
        image = Image.open(image_file_path)
        exif_data = image._getexif()
        if not exif_data:
            print("Nessun dato EXIF trovato.")
            return None

        readable_exif = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
        x_res = readable_exif.get("FocalPlaneXResolution")
        res_unit = readable_exif.get("FocalPlaneResolutionUnit")

        if not x_res or not res_unit:
            print("Dati di risoluzione del piano focale mancanti.")
            return None

        if res_unit == 2:  # Inch
            pixel_size_um = (1 / x_res) * 25.4 * 1000
        elif res_unit == 3:  # Centimetri
            pixel_size_um = (1 / x_res) * 10 * 1000
        else:
            print(f"Unità di misura non supportata (code: {res_unit})")
            return None

        pixel_area_um2 = pixel_size_um ** 2

        return {
            "pixel_size_um": pixel_size_um,
            "pixel_area_um2": pixel_area_um2
        }

    except Exception as e:
        print(f"Errore durante la lettura dell'immagine: {e}")
        return None


# =========================
# ===== Utilità COCO ======
# =========================

def load_contiguous_class_names(coco_path: Path) -> List[str]:
    """Restituisce i nomi classe indicizzati come il modello (0=background)."""
    if not coco_path.exists():
        return ["background"] + [f"class_{i}" for i in range(1, NUM_CLASSES)]

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = coco.get("categories", [])
    cats_sorted = sorted(cats, key=lambda c: c["id"])
    names = ["background"] + [c.get("name", f"class_{i+1}") for i, c in enumerate(cats_sorted)]

    if len(names) != NUM_CLASSES:
        names = names[:NUM_CLASSES] if len(names) >= NUM_CLASSES else names + [
            f"class_{i}" for i in range(len(names), NUM_CLASSES)
        ]
    return names


# =========================
# ===== Modello / Vis =====
# =========================

def build_model(num_classes: int) -> torch.nn.Module:
    """Crea Mask R-CNN con classificatori sostituiti per num_classes."""
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    return model


def overlay_fill_only(
    image_np: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    score_thr: float,
    area_info: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Applica il riempimento delle maschere e stampa l’area."""
    h, w = image_np.shape[:2]
    total_px = float(h * w)

    kept = scores >= score_thr
    masks = masks[kept]
    scores = scores[kept]
    labels = labels[kept]

    if len(masks) == 0:
        print("— Nessuna maschera sopra soglia —")
        return image_np, masks, labels, scores

    order = np.argsort(scores)[::-1]
    masks = masks[order]
    scores = scores[order]
    labels = labels[order]

    img_out = image_np.copy()
    cmap = plt.get_cmap("tab20")

    for i in range(len(masks)):
        color_rgb = np.array(cmap(i % cmap.N))[:3]
        mask_bin = masks[i] > 0.5

        area_px = int(mask_bin.sum())
        area_pct = (area_px / total_px) * 100.0
        cls_idx = int(labels[i])
        cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"id_{cls_idx}"
        print(f"   • {cls_name}: score={scores[i]:.2f}  area={area_px*area_info['pixel_area_um2']} µm²  ({area_pct:.2f}%)")

        img_out[mask_bin] = (
            img_out[mask_bin] * (1.0 - FILL_ALPHA) + color_rgb * 255.0 * FILL_ALPHA
        ).astype(np.uint8)

    return img_out, masks, labels, scores


def show_and_wait(
    img_np: np.ndarray,
    window_title: str,
    masks: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    class_names: List[str] | None = None,
    area_info: dict | None = None,
) -> bool:
    """Mostra immagine con solo box di testo sopra gli oggetti, senza contorni."""
    want_exit = {"flag": False}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np)
    ax.axis("off")
    fig.suptitle(window_title, fontsize=12)
    fig.tight_layout()

    if masks is not None and len(masks) > 0:
        for i in range(len(masks)):
            z = (masks[i] > 0.5).astype(float)
            ys, xs = np.nonzero(z > 0.5)
            if len(xs) > 0:
                # Centroide
                cx, cy = int(xs.mean()), int(ys.mean())

                # Area in µm² o px
                area_px = len(xs)
                if area_info is not None:
                    area_um2 = area_px * area_info["pixel_area_um2"]
                    area_str = f"{area_um2:.0f} µm²"
                else:
                    area_str = f"{area_px} px"

                # Testo sopra il centroide
                cls_idx = int(labels[i]) if labels is not None else -1
                cls_name = (
                    class_names[cls_idx]
                    if (class_names is not None and 0 <= cls_idx < len(class_names))
                    else f"id_{cls_idx}"
                )
                score_str = f"{float(scores[i]):.2f}" if scores is not None else ""
                ax.text(
                    cx, cy - 10,  # leggermente sopra il centroide
                    f"{cls_name} {score_str}\n{area_str}",
                    ha="center", va="bottom",
                    fontsize=10, color="white",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2),
                )

    def on_key(event):
        key = (event.key or "").lower()
        if key in ("escape", "esc", "q"):
            want_exit["flag"] = True
        plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return want_exit["flag"]


# =========================
# ========== Main =========
# =========================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modello non trovato: {MODEL_PATH}")

    image_paths = sorted(
        [*IMAGE_DIR.glob("*.jpg"), *IMAGE_DIR.glob("*.jpeg"), *IMAGE_DIR.glob("*.png")]
    )
    if not image_paths:
        raise FileNotFoundError(f"Nessuna immagine trovata in: {IMAGE_DIR.resolve()}")

    model = build_model(NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    class_names = load_contiguous_class_names(COCO_PATH)

    print(f"➡️  Trovate {len(image_paths)} immagini. Premi un tasto per avanzare; ESC/q per uscire.\n")

    for img_path in image_paths:
        print(f"\n📷 {img_path.name}")
        info = get_pixel_info(img_path)
        image_pil = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)[0]

        masks = output["masks"].cpu().numpy()[:, 0, ...]
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        img_np = np.array(image_pil)
        img_filled, kept_masks, kept_labels, kept_scores = overlay_fill_only(
            image_np=img_np,
            masks=masks,
            scores=scores,
            labels=labels,
            class_names=class_names,
            score_thr=SCORE_THRESHOLD,
            area_info=info if info else {"pixel_area_um2": 1.0},
        )

        title = f"{img_path.name} — premi un tasto per continuare, ESC/q per uscire"
        if show_and_wait(
            img_np=img_filled,
            window_title=title,
            masks=kept_masks,
            labels=kept_labels,
            scores=kept_scores,
            class_names=class_names,
            area_info=info if info else None,
        ):
            print("👋 Uscita richiesta dall'utente.")
            break


if __name__ == "__main__":
    main()
