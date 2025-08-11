#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch converter: Labelme JSON -> COCO (uno-a-uno)

- Legge tutti i file .json in "labeled img/"
- Converte ciascun file Labelme in un COCO autonomo
- Salva in "coco/<stem>_coco.json"
- Usa file_name = basename dell'immagine (senza cartelle)
- Gestione robusta di image size (imageHeight/Width, file sul disco, imageData)

Limitazioni:
- Converte solo shape di tipo "polygon". Altri tipi vengono ignorati con warning.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image


# =========================
# ===== CONFIG PATHS ======
# =========================

LABELED_DIR = Path("labeled img")   # dove stanno i JSON Labelme
UNLABELED_DIR = Path("unlabeled img")  # dove potrebbero stare le immagini
OUTPUT_DIR = Path("coco")           # dove salvare i COCO risultanti


# =========================
# ===== UTILITIES =========
# =========================

def resolve_image_size(
    lm_dict: Dict,
    json_path: Path,
    unlabeled_dir: Path
) -> Tuple[int, int, str]:
    """
    Determina (width, height) e il file_name (basename) dell'immagine.
    Ordine tentativi:
      1) usa imageWidth/imageHeight se presenti
      2) prova ad aprire imagePath come percorso assoluto/relativo
      3) prova ad aprire basename(imagePath) dentro unlabeled_dir
      4) se presente imageData (base64), ricava dimensioni da lì

    Ritorna: (width, height, file_name_basename)
    Lancia ValueError se impossibile determinare.
    """
    image_path_field = (lm_dict.get("imagePath") or "").strip()
    file_name_basename = Path(image_path_field).name if image_path_field else ""

    width = lm_dict.get("imageWidth")
    height = lm_dict.get("imageHeight")
    if isinstance(width, int) and isinstance(height, int):
        if not file_name_basename:
            raise ValueError(f"imagePath mancante in {json_path.name}")
        return width, height, file_name_basename

    # Prova ad aprire il percorso indicato in imagePath (relativo al JSON)
    if image_path_field:
        candidate = (json_path.parent / image_path_field).resolve()
        if candidate.exists():
            with Image.open(candidate) as im:
                w, h = im.size
            return w, h, candidate.name

    # Prova a cercare l'immagine per basename dentro unlabeled_dir
    if file_name_basename:
        candidate2 = (unlabeled_dir / file_name_basename).resolve()
        if candidate2.exists():
            with Image.open(candidate2) as im:
                w, h = im.size
            return w, h, candidate2.name

    # Estrema ratio: se c'è imageData base64, apri da lì
    image_data_b64 = lm_dict.get("imageData")
    if image_data_b64:
        try:
            raw = base64.b64decode(image_data_b64)
            with Image.open(io.BytesIO(raw)) as im:
                w, h = im.size
            # Se mancava imagePath, crea un nome fittizio ma stabile
            if not file_name_basename:
                file_name_basename = f"{json_path.stem}.png"
            return w, h, file_name_basename
        except Exception as exc:
            raise ValueError(
                f"Impossibile ricavare dimensioni da imageData in {json_path.name}: {exc}"
            ) from exc

    raise ValueError(
        f"Impossibile determinare dimensioni immagine per {json_path.name}. "
        f"Aggiungi imageWidth/Height, un imagePath valido, o imageData."
    )


def polygon_to_bbox(segmentation: List[float]) -> Tuple[float, float, float, float, float]:
    """
    Calcola bbox (x, y, w, h) e area a partire da una segmentation poligonale (x1,y1,x2,y2,...).
    """
    xs = segmentation[0::2]
    ys = segmentation[1::2]
    x_min = float(min(xs))
    y_min = float(min(ys))
    x_max = float(max(xs))
    y_max = float(max(ys))
    w = x_max - x_min
    h = y_max - y_min
    area = float(w * h)
    return x_min, y_min, w, h, area


def labelme_to_coco_single(lm_dict: Dict, json_path: Path, unlabeled_dir: Path) -> Dict:
    """
    Converte un singolo dizionario Labelme in un dizionario COCO (per 1 immagine).
    - Crea categories in base alle label presenti in questo JSON.
    - Include solo shapes di tipo 'polygon'.
    """
    width, height, file_name = resolve_image_size(lm_dict, json_path, unlabeled_dir)

    coco: Dict = {
        "images": [{
            "id": 0,
            "file_name": file_name,  # uso solo il basename
            "width": width,
            "height": height
        }],
        "annotations": [],
        "categories": []
    }

    label_to_id: Dict[str, int] = {}
    next_cat_id = 1
    next_ann_id = 1

    shapes = lm_dict.get("shapes") or []
    for shape in shapes:
        shape_type = shape.get("shape_type", "polygon")
        if shape_type != "polygon":
            # salta gli shape non poligonali
            print(f"[WARN] {json_path.name}: shape '{shape_type}' ignorato (solo polygon supportato).")
            continue

        label = str(shape.get("label", "")).strip() or "unknown"
        points = shape.get("points") or []
        if len(points) < 3:
            print(f"[WARN] {json_path.name}: poligono con meno di 3 punti ignorato.")
            continue

        # segmentation: flat list [x1,y1,x2,y2,...]
        flat = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                continue
            flat.extend([float(pt[0]), float(pt[1])])

        if len(flat) < 6:
            print(f"[WARN] {json_path.name}: poligono non valido ignorato.")
            continue

        if label not in label_to_id:
            label_to_id[label] = next_cat_id
            coco["categories"].append({
                "id": next_cat_id,
                "name": label,
                "supercategory": "none"
            })
            next_cat_id += 1

        x_min, y_min, w, h, area = polygon_to_bbox(flat)

        coco["annotations"].append({
            "id": next_ann_id,
            "image_id": 0,
            "category_id": label_to_id[label],
            "segmentation": [flat],
            "bbox": [x_min, y_min, w, h],
            "area": area,
            "iscrowd": 0
        })
        next_ann_id += 1

    return coco


# =========================
# ===== MAIN (batch) ======
# =========================

def main() -> None:
    """
    Converte tutti i JSON in LABELED_DIR e salva i COCO in OUTPUT_DIR.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(LABELED_DIR.glob("*.json"))
    if not json_files:
        print(f"[INFO] Nessun .json trovato in: {LABELED_DIR.resolve()}")
        return

    converted = 0
    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                lm_dict = json.load(f)
        except Exception as exc:
            print(f"[SKIP] Impossibile leggere {json_path.name}: {exc}")
            continue

        # Riconosce Labelme: deve avere almeno shapes + imagePath o imageData
        if not isinstance(lm_dict, dict) or "shapes" not in lm_dict:
            print(f"[SKIP] {json_path.name} non sembra Labelme (manca 'shapes').")
            continue

        try:
            coco_dict = labelme_to_coco_single(lm_dict, json_path, UNLABELED_DIR)
        except ValueError as exc:
            print(f"[SKIP] {json_path.name}: {exc}")
            continue

        out_name = f"{json_path.stem}_coco.json"
        out_path = OUTPUT_DIR / out_name

        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(coco_dict, f, indent=2, ensure_ascii=False)
            converted += 1
            print(f"[OK] {json_path.name} → {out_path}")
        except Exception as exc:
            print(f"[ERR] Non riesco a scrivere {out_path.name}: {exc}")

    print(f"\n✅ Conversione completata. File convertiti: {converted}. "
          f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
