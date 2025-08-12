#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data augmentation per un dataset COCO con:
- clipping dei poligoni
- ID univoci (images/annotations) nel COCO aggregato
- normalizzazione categorie per NOME -> category_id consistente

Legge immagini da "unlabeled img/" e i COCO per-immagine da "coco/".
Scrive immagini aumentate + COCO aggregato in "augmented/".
"""

import json
import math
import random
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# =========================
# ========= CONFIG ========
# =========================

UNLABELED_DIR = Path("unlabeled img")
LABELED_DIR   = Path("coco")
OUTPUT_DIR    = Path("augmented")

NUM_COPIES_PER_IMAGE = 10
RANDOM_SEED: Optional[int] = None

ROTATION_ANGLES  = [0, 45, 90, 135, 180, 225, 270, 315]
SCALE_RANGE      = (0.8, 1.2)
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE   = (0.7, 1.3)

# Filtri anti-degeneri
MIN_BOX_SIDE_PX = 1.0     # scarta bbox con lato <= 1 px
MIN_POLY_AREA   = 5.0     # scarta poligoni con area |A| < 5 px^2


# =========================
# ===== SUPPORTO COCO =====
# =========================

def load_coco_for_image(image_path: Path) -> Optional[Tuple[Dict, Dict, List[Dict]]]:
    """
    Carica il JSON COCO relativo a un'immagine, se esiste.
    Cerca: {stem}_coco.json oppure {stem}.json in LABELED_DIR.
    Ritorna (coco_dict, image_entry, annotations) o None se non trovato.
    """
    stem = image_path.stem
    candidates = [
        LABELED_DIR / f"{stem}_coco.json",
        LABELED_DIR / f"{stem}.json",
    ]

    for json_path in candidates:
        if not json_path.exists():
            continue
        try:
            with json_path.open("r", encoding="utf-8") as f:
                coco = json.load(f)
        except Exception:
            continue

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])

        if len(images) == 1:
            img_entry = images[0]
        else:
            img_entry = next(
                (img for img in images if img.get("file_name") == image_path.name),
                None
            )
            if img_entry is None:
                continue

        img_id = img_entry.get("id")
        anns = [a for a in annotations if a.get("image_id") == img_id]
        return coco, img_entry, anns

    return None


# =========================
# ====== TRASFORMAZIONI ===
# =========================

def apply_geom_to_coords(coords: np.ndarray, angle: float, flip: bool, size: Tuple[int, int]) -> np.ndarray:
    """Applica rotazione (circa il centro) e flip orizzontale a coordinate (x,y)."""
    w, h = size
    # porta l'origine in basso a sinistra -> ruota -> riporta
    coords[:, 1] = h - coords[:, 1]
    coords -= np.array([w / 2, h / 2], dtype=np.float32)

    theta = math.radians(angle)
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta),  math.cos(theta)]], dtype=np.float32)
    coords = coords @ rot.T

    coords += np.array([w / 2, h / 2], dtype=np.float32)
    coords[:, 1] = h - coords[:, 1]

    if flip:
        coords[:, 0] = w - coords[:, 0]

    return coords


def transform_image_and_polygons(
    img: Image.Image,
    polygons: List[List[float]],
    angle: float,
    flip: bool,
    scale: float
) -> Tuple[Image.Image, List[List[float]]]:
    """Applica scala, rotazione e flip a immagine e poligoni."""
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)

    # scala + crop centrale per mantenere dimensioni finali == originali
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - w) // 2
    top  = (new_h - h) // 2
    img_cropped = img_resized.crop((left, top, left + w, top + h))

    # rotazione + flip immagine
    img_rotated = TF.rotate(img_cropped, angle)  # expand=False
    if flip:
        img_rotated = TF.hflip(img_rotated)

    # trasformazioni poligoni (stesso ordine: scala intorno al centro + rot/flip)
    new_polys = []
    for poly in polygons:
        coords = np.array(poly, dtype=np.float32).reshape(-1, 2)
        coords = (coords - np.array([w / 2, h / 2], dtype=np.float32)) * scale + np.array([w / 2, h / 2], dtype=np.float32)
        coords = apply_geom_to_coords(coords, angle, flip, (w, h))
        new_polys.append(coords.flatten().tolist())

    return img_rotated, new_polys


# =========================
# ==== CLIP & VALIDATE ====
# =========================

def polygon_shoelace_area(xs: np.ndarray, ys: np.ndarray) -> float:
    """Area con formula di Gauss (shoelace). Ritorna |area|."""
    if xs.size < 3:
        return 0.0
    x_shift = np.roll(xs, -1)
    y_shift = np.roll(ys, -1)
    area = 0.5 * np.abs(np.sum(xs * y_shift - x_shift * ys))
    return float(area)

def clip_polygon_to_image(poly: List[float], w: int, h: int) -> Optional[List[float]]:
    """
    Clippa un poligono ai bordi [0..w-1],[0..h-1].
    Elimina poligoni con <3 vertici distinti o area troppo piccola.
    Ritorna la lista piatta [x1,y1,...] clippata, oppure None se da scartare.
    """
    arr = np.array(poly, dtype=np.float32).reshape(-1, 2)
    # clip
    arr[:, 0] = np.clip(arr[:, 0], 0.0, w - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, h - 1)

    # rimuovi vertici duplicati consecutivi
    if arr.shape[0] >= 2:
        keep = [0]
        for i in range(1, arr.shape[0]):
            if not np.allclose(arr[i], arr[keep[-1]]):
                keep.append(i)
        arr = arr[keep]

    if arr.shape[0] < 3:
        return None

    xs, ys = arr[:, 0], arr[:, 1]
    # bbox rapida per degeneri
    w_box = float(xs.max() - xs.min())
    h_box = float(ys.max() - ys.min())
    if w_box <= MIN_BOX_SIDE_PX or h_box <= MIN_BOX_SIDE_PX:
        return None

    # area del poligono
    area = polygon_shoelace_area(xs, ys)
    if area < MIN_POLY_AREA:
        return None

    return arr.flatten().tolist()


# =========================
# ===== MANAGE OUTPUT =====
# =========================

def reset_output_dir(output_dir: Path) -> None:
    """Pulisce e ricrea la cartella di output (con guardia sul nome)."""
    if output_dir.exists():
        if output_dir.is_dir() and output_dir.name == "augmented":
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(f"Percorso output sospetto: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


# =========================
# ========= MAIN ==========
# =========================

def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    reset_output_dir(OUTPUT_DIR)

    # COCO aggregato
    aggregated_coco: Dict = {"images": [], "annotations": [], "categories": []}

    # Mapping categorie per NOME -> id aggregato
    cat_name_to_aggid: Dict[str, int] = {}
    next_cat_id = 1

    # contatori univoci per immagini/annotazioni
    global_img_id = 1
    global_ann_id = 1

    image_files = sorted(
        [*UNLABELED_DIR.glob("*.png"),
         *UNLABELED_DIR.glob("*.jpg"),
         *UNLABELED_DIR.glob("*.jpeg")]
    )

    for img_path in image_files:
        result = load_coco_for_image(img_path)
        if result is None:
            print(f"[SKIP] Nessun COCO trovato per {img_path.name}")
            continue

        coco, img_entry, anns = result

        # mapping ID->nome categorie nel COCO sorgente
        src_cats = coco.get("categories", []) or []
        src_id_to_name = {c["id"]: str(c.get("name", "")).strip() for c in src_cats}

        # assicurati che tutte le categorie incontrate abbiano un id aggregato
        for name in src_id_to_name.values():
            if not name:
                name = "unknown"
            if name not in cat_name_to_aggid:
                cat_name_to_aggid[name] = next_cat_id
                aggregated_coco["categories"].append({
                    "id": next_cat_id,
                    "name": name,
                    "supercategory": "none"
                })
                next_cat_id += 1

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # prendi tutte le segmentation dell'immagine
        segmentations = []
        category_ids = []
        for a in anns:
            seg = a.get("segmentation")
            if not seg:
                continue
            segmentations.append(seg[0])
            category_ids.append(a.get("category_id"))

        used_combos: set = set()
        for i in range(NUM_COPIES_PER_IMAGE):
            # === sampling parametri augmentation (unici per immagine) ===
            for _ in range(1000):
                combo = (
                    random.choice(ROTATION_ANGLES),
                    random.choice([True, False]),
                    round(random.uniform(*SCALE_RANGE), 3),
                    round(random.uniform(*BRIGHTNESS_RANGE), 3),
                    round(random.uniform(*CONTRAST_RANGE), 3),
                )
                if combo not in used_combos:
                    used_combos.add(combo)
                    break
            else:
                print(f"[STOP] {img_path.stem}: troppe combinazioni duplicate.")
                break

            angle, flip, scale, bright, contrast = combo

            # === applica trasformazioni ===
            aug_img, aug_polys = transform_image_and_polygons(img, segmentations, angle, flip, scale)
            aug_img = TF.adjust_brightness(aug_img, bright)
            aug_img = TF.adjust_contrast(aug_img, contrast)

            out_name = f"{img_path.stem}_aug_{i:03d}.png"
            aug_img.save(OUTPUT_DIR / out_name)

            # IMMAGINE con ID UNIVOCO
            aggregated_coco["images"].append({
                "id": global_img_id,
                "file_name": out_name,
                "width": w,
                "height": h
            })

            dropped = 0
            kept = 0

            # === crea annotazioni clippate ===
            for j, poly in enumerate(aug_polys):
                clipped = clip_polygon_to_image(poly, w, h)
                if clipped is None:
                    dropped += 1
                    continue

                xs = np.array(clipped[0::2], dtype=np.float32)
                ys = np.array(clipped[1::2], dtype=np.float32)

                x_min, y_min = float(xs.min()), float(ys.min())
                w_box = float(xs.max() - x_min)
                h_box = float(ys.max() - y_min)
                if w_box <= MIN_BOX_SIDE_PX or h_box <= MIN_BOX_SIDE_PX:
                    dropped += 1
                    continue

                poly_area = max(1.0, polygon_shoelace_area(xs, ys))  # area >= 1

                # mappa category_id di origine -> nome -> id aggregato
                src_cid = category_ids[j] if j < len(category_ids) else None
                cat_name = src_id_to_name.get(src_cid, "unknown")
                if cat_name not in cat_name_to_aggid:
                    # (di norma non accade, ma per sicurezza assegna un nuovo id)
                    cat_name_to_aggid[cat_name] = next_cat_id
                    aggregated_coco["categories"].append({
                        "id": next_cat_id,
                        "name": cat_name,
                        "supercategory": "none"
                    })
                    next_cat_id += 1
                agg_cid = cat_name_to_aggid[cat_name]

                # ANNOTAZIONE con ID UNIVOCO e category_id aggregato
                aggregated_coco["annotations"].append({
                    "id": global_ann_id,
                    "image_id": global_img_id,
                    "category_id": agg_cid,
                    "segmentation": [clipped],
                    "bbox": [x_min, y_min, w_box, h_box],
                    "area": poly_area,
                    "iscrowd": 0
                })
                global_ann_id += 1
                kept += 1

            print(f"[OK] {img_path.stem} aug#{i:03d} → kept={kept}  dropped={dropped}")
            global_img_id += 1

        print(f"[OK] {img_path.stem} processata.")

    # (opzionale) ordina categories per id per stabilità
    aggregated_coco["categories"].sort(key=lambda c: c["id"])

    with open(OUTPUT_DIR / "augmented_coco.json", "w", encoding="utf-8") as f:
        json.dump(aggregated_coco, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Augmentation completata. File COCO: {OUTPUT_DIR / 'augmented_coco.json'}")


if __name__ == "__main__":
    main()
