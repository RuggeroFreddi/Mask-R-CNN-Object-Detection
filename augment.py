#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data augmentation per un dataset COCO:
- Legge tutte le immagini da "unlabeled img/"
- Usa solo quelle che hanno un relativo file COCO in "coco/"
- Applica trasformazioni geometriche e fotometriche alle immagini e alle
  annotazioni poligonali.
- Pulisce e ricrea "augmented/" e salva lì immagini aumentate + COCO aggregato.
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
LABELED_DIR = Path("coco")
OUTPUT_DIR = Path("augmented")

NUM_COPIES_PER_IMAGE = 10
RANDOM_SEED: Optional[int] = None

ROTATION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
SCALE_RANGE = (0.8, 1.2)
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.7, 1.3)


# =========================
# ===== SUPPORTO COCO =====
# =========================

def load_coco_for_image(image_path: Path) -> Optional[Tuple[Dict, Dict, List[Dict]]]:
    """
    Carica il JSON COCO relativo a un'immagine, se esiste.
    Cerca:
      - {stem}_coco.json
      - {stem}.json
    Restituisce (coco_dict, image_entry, annotations) o None se non trovato.
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


def validate_categories(base_cats: Optional[List[Dict]], new_cats: List[Dict]) -> List[Dict]:
    """Mantiene coerenza delle categorie tra diversi COCO."""
    if base_cats is None:
        return deepcopy(new_cats)
    base_map = {c["id"]: c.get("name") for c in base_cats}
    for cat in new_cats:
        if base_map.get(cat["id"]) != cat.get("name"):
            print(f"[AVVISO] Category ID {cat['id']} ha nome diverso nei file.")
    return base_cats


# =========================
# ====== TRASFORMAZIONI ===
# =========================

def apply_geom_to_coords(coords: np.ndarray, angle: float, flip: bool, size: Tuple[int, int]) -> np.ndarray:
    """Applica rotazione e flip a coordinate (x, y) mantenendo il centro immagine."""
    w, h = size
    coords[:, 1] = h - coords[:, 1]
    coords -= np.array([w / 2, h / 2])

    theta = math.radians(angle)
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta),  math.cos(theta)]])
    coords = coords @ rot.T
    coords += np.array([w / 2, h / 2])
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

    img_resized = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    img_cropped = img_resized.crop((left, top, left + w, top + h))

    img_rotated = TF.rotate(img_cropped, angle)  # expand=False per default
    if flip:
        img_rotated = TF.hflip(img_rotated)

    new_polys = []
    for poly in polygons:
        coords = np.array(poly).reshape(-1, 2)
        coords = (coords - np.array([w / 2, h / 2])) * scale + np.array([w / 2, h / 2])
        coords = apply_geom_to_coords(coords, angle, flip, (w, h))
        new_polys.append(coords.flatten().tolist())

    return img_rotated, new_polys


def sample_unique_combo(used: set) -> Tuple[int, bool, float, float, float]:
    """Estrae una combinazione random non già usata."""
    for _ in range(1000):
        combo = (
            random.choice(ROTATION_ANGLES),
            random.choice([True, False]),
            round(random.uniform(*SCALE_RANGE), 3),
            round(random.uniform(*BRIGHTNESS_RANGE), 3),
            round(random.uniform(*CONTRAST_RANGE), 3),
        )
        if combo not in used:
            used.add(combo)
            return combo
    raise RuntimeError("Troppe combinazioni duplicate.")


# =========================
# ===== MANAGE OUTPUT =====
# =========================

def reset_output_dir(output_dir: Path) -> None:
    """
    Se esiste, cancella completamente la cartella di output e la ricrea.
    Include una piccola guardia per evitare cancellazioni indesiderate.
    """
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

    # Pulisci e ricrea la cartella di output
    reset_output_dir(OUTPUT_DIR)

    aggregated_coco: Dict = {"images": [], "annotations": [], "categories": []}
    base_categories: Optional[List[Dict]] = None
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
        base_categories = validate_categories(base_categories, coco.get("categories", []))
        aggregated_coco["categories"] = base_categories

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        segmentations = [a["segmentation"][0] for a in anns if a.get("segmentation")]
        category_ids = [a["category_id"] for a in anns if a.get("segmentation")]

        used_combos: set = set()
        for i in range(NUM_COPIES_PER_IMAGE):
            try:
                angle, flip, scale, bright, contrast = sample_unique_combo(used_combos)
            except RuntimeError as e:
                print(f"[STOP] {img_path.stem}: {e}")
                break

            aug_img, aug_polys = transform_image_and_polygons(img, segmentations, angle, flip, scale)
            aug_img = TF.adjust_brightness(aug_img, bright)
            aug_img = TF.adjust_contrast(aug_img, contrast)

            out_name = f"{img_path.stem}_aug_{i:03d}.png"
            aug_img.save(OUTPUT_DIR / out_name)

            aggregated_coco["images"].append({
                "id": global_img_id,
                "file_name": out_name,
                "width": w,
                "height": h
            })

            for j, poly in enumerate(aug_polys):
                xs, ys = poly[0::2], poly[1::2]
                x_min, y_min = min(xs), min(ys)
                w_box, h_box = max(xs) - x_min, max(ys) - y_min
                if w_box <= 0 or h_box <= 0:
                    continue
                aggregated_coco["annotations"].append({
                    "id": global_ann_id,
                    "image_id": global_img_id,
                    "category_id": category_ids[j] if j < len(category_ids) else 1,
                    "segmentation": [poly],
                    "bbox": [x_min, y_min, w_box, h_box],
                    "area": w_box * h_box,
                    "iscrowd": 0
                })
                global_ann_id += 1

            global_img_id += 1

        print(f"[OK] {img_path.stem} processata.")

    with open(OUTPUT_DIR / "augmented_coco.json", "w", encoding="utf-8") as f:
        json.dump(aggregated_coco, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Augmentation completata. File COCO: {OUTPUT_DIR / 'augmented_coco.json'}")


if __name__ == "__main__":
    main()
