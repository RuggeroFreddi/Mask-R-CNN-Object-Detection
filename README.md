# README — Istruzioni per addestrare e testare Mask R-CNN (ResNet-50)

> Pipeline completa: etichettatura con **Labelme**, conversione **Labelme → COCO**, **data augmentation**, **training** (5 epoche di default), **test** con visualizzazione maschera e stampa dell’area in pixel.

---

## Requisiti rapidi

- **Python** ≥ 3.9
- (Consigliato) **GPU NVIDIA** con driver + CUDA
- Pacchetti principali (se non esiste `requirements.txt`):
  ```bash
  pip install labelme
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   # per GPU CUDA 12.1 (adatta alla tua macchina)
  pip install numpy opencv-python pillow matplotlib tqdm pycocotools
  ```

> Su Windows, se `pycocotools` dà errore:
> ```bash
> pip install cython
> pip install "pycocotools-windows; platform_system == 'Windows'"
> ```

---

## Struttura cartelle attesa

```
project/
├─ unlabeled img/         # immagini grezze da etichettare
├─ labeled img/           # JSON Labelme salvati da Labelme
├─ coco/                  # JSON in formato COCO (+ eventuali immagini)
├─ test img/              # immagini per il test del modello
├─ to coco.py
├─ augment.py
├─ train model.py
└─ test model.py
```

> I percorsi contengono **spazi**: usa le **virgolette** quando richiesto (es. `"labeled img"`). In alternativa rinomina senza spazi e aggiorna gli script.

---

## 1) Etichettare le immagini (Labelme)

Apri un terminale nella cartella del progetto e lancia:

```bash
labelme
```

- Carica le immagini da **`unlabeled img/`**.
- Disegna poligoni/maschere e assegna le **etichette**.
- Salva i file **`.json`** in **`labeled img/`** (formato *json-labelme*).

Consigli:
- Usa **nomi classe coerenti** (case-sensitive).
- Chiudi i poligoni con cura per maschere pulite.

---

## 2) Conversione Labelme → COCO

Esegui lo script di conversione (**salva i nuovi JSON in `coco/`**):

```bash
python "to coco.py"
```

> Se lo script prevede parametri/percorsi, apri `to coco.py` e verifica costanti come:
> ```python
> LABELME_DIR = "labeled img"
> COCO_OUT   = "coco/annotations.json"
> IMAGES_DIR = "coco/images"  # se gestisci una copia/riordino delle immagini
> ```

---

## 3) Data augmentation

Genera nuove coppie immagine+annotazione con trasformazioni **casuali**:

```bash
python "augment.py"
```

### Trasformazioni effettivamente usate
> **Sostituisci/aggiorna questa lista in base a ciò che fa davvero `augment.py`.** Esempio tipico:
- Flip orizzontale
- Flip verticale
- Rotazione casuale (±X°)
- Ridimensionamento / Zoom
- Traslazione (shift X/Y)
- Crop casuale e/o padding
- Shear (taglio)
- Jitter colore (luminosità/contrasto/saturazione/tono)
- Rumore gaussiano
- Blur (gaussiano/median)
- Deformazione elastica
- Cutout / Random erasing
- (Opzionale) Prospettiva / Affine

> Verifica anche: probabilità per trasformazione, range dei parametri e **quante immagini aggiuntive** vengono create.

---

## 4) Addestramento del modello (Mask R-CNN, backbone ResNet-50)

Avvia il training (di default **5 epoche**):

```bash
python "train model.py"
```

Punti da controllare nello script:
- **Numero epoche** (aumenta quando avrai più dati):
  ```python
  EPOCHS = 5  # aumenta se il dataset cresce
  ```
- **Dataloader COCO**: percorsi al JSON e alle immagini in `coco/`.
- **Dispositivo**: usa `cuda()` se disponibile (`torch.cuda.is_available()`).
- **Salvataggio modello**: verifica il percorso del file (`.pth`, `.pt`).

Suggerimenti:
- Se la GPU esaurisce memoria: riduci `BATCH_SIZE`, risoluzione, o usa gradient accumulation (se presente).
- Monitora loss e metriche; se vedi underfitting, aumenta epoche/learning rate (con cautela).

---

## 5) Test del modello

Esegui il test sulle immagini in **`test img/`**:

```bash
python "test model.py"
```

Comportamento atteso per **ogni immagine** nella cartella:
- Visualizzazione della **maschera** dell’oggetto trovato (overlay o binaria).
- Stampa in console dell’**area in pixel** della maschera rilevata, es.:
  ```
  image_001.jpg → area: 15234 px
  image_002.jpg → area: 9876 px
  ```

> Se desideri anche **salvare** le immagini con overlay, aggiungi in `test model.py` qualcosa come:
> ```python
> cv2.imwrite("outputs/overlay_image_001.jpg", overlay)
> ```

---

## Troubleshooting

- **Nessuna maschera / errori in training** → Verifica che i JSON COCO abbiano `categories`, `images`, `annotations` corretti e che i nomi classe siano identici a quelli usati in Labelme.
- **Percorsi con spazi** → Usa virgolette o rinomina le cartelle.
- **pycocotools su Windows** → Vedi nota in *Requisiti rapidi*.
- **Lentezza / out-of-memory** → Riduci batch size o risoluzione; valuta il numero di workers del DataLoader.

---

## Comandi rapidi (riassunto)

```bash
# 1) Etichetta con Labelme
labelme   # salva i .json in "labeled img"

# 2) Converti a COCO
python "to coco.py"

# 3) Augmenta i dati
python "augment.py"

# 4) Allena (5 epoche di default nello script)
python "train model.py"

# 5) Testa sul set di test
python "test model.py"
```

> Questo README è allineato alla descrizione fornita. Se cambi nomi/cartelle/parametri negli script, **ricordati di aggiornare il documento**.

