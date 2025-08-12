# Mask R-CNN — Blocchi principali e iperparametri (torchvision)

## Come funziona (panoramica super rapida)
**Mask R-CNN** ha quattro blocchi principali:
- **ResNet (backbone)**: estrae feature dall’immagine.
- **FPN**: fonde feature a più scale (piramide) per catturare oggetti piccoli/grandi.
- **RPN**: scansiona le feature con **anchor** e propone **RoI** (region proposals) agnostiche alla classe.
- **RoI (RoIAlign + heads)**: RoIAlign ritaglia le feature per ciascuna proposta; la **box head** classifica + rifinisce le bbox, la **mask head** produce la **maschera** per la classe.

---

## Iperparametri principali (cosa fanno + esempi)
> ⚡ = valori/assetti tipici predefiniti o comunemente usati

### 1) Backbone (ResNet) + Transform
- **`weights`** — inizializzazione (pretrain).  
  *Esempi*: `DEFAULT`⚡ (più stabile), `None` (da zero).
- **`trainable_backbone_layers`** — quanti stadi del backbone si allenano.  
  *Esempi*: `3`⚡ (ultimi stadi), `5` (fine-tune completo).
- **`model.transform.min_size` / `max_size`** — resize lato corto/lungo in input.  
  *Esempi*: `min_size=(800,)`⚡, `max_size=1333`⚡ → bilanciato; `min_size=(640,)`, `max_size=1024` → più veloce/meno memoria.
- **`image_mean` / `image_std`** — normalizzazione input.  
  *Esempi*: ImageNet⚡ (`[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`); personalizzati se cambi preprocessing.

### 2) FPN (Feature Pyramid Network)
- **Livelli FPN (P2–P5/P6)** — quante scale di feature usi.  
  *Esempi*: P2–P5⚡ (standard); aggiungi P6 per oggetti molto grandi.
- **`out_channels`** — canali delle mappe FPN.  
  *Esempi*: `256`⚡ (standard), `128` (meno memoria, possibile calo).

### 3) RPN (Region Proposal Network)
- **`anchor_generator.sizes`** — scale delle anchor per livello.  
  *Esempi*: `((32,64,128,256,512),)`⚡; `((8,16,32,64,128),)` per oggetti piccoli.
- **`anchor_generator.aspect_ratios`** — rapporti d’aspetto delle anchor.  
  *Esempi*: `((0.5,1.0,2.0),)`⚡; `((0.33,0.5,1.0,2.0,3.0),)` per forme estreme.
- **`rpn_fg_iou_thresh` / `rpn_bg_iou_thresh`** — soglie IoU per etichettare anchor fg/bg.  
  *Esempi*: `0.7 / 0.3`⚡ (selettivo); `0.6 / 0.2` (più tollerante).
- **`rpn_batch_size_per_image`** — campioni/immagine per la loss RPN.  
  *Esempi*: `256`⚡; `512` (più segnale, più memoria).
- **`rpn_positive_fraction`** — quota positivi nel batch RPN.  
  *Esempi*: `0.5`⚡; `0.25` se pochi positivi rumorosi.
- **`pre_nms_top_n_train/test`** — proposte tenute **prima** della NMS.  
  *Esempi*: `2000/1000`⚡; `600/300` per accelerare.
- **`post_nms_top_n_train/test`** — proposte **dopo** NMS.  
  *Esempi*: `1000/1000`⚡; `300/300` per meno RoI.
- **`rpn.nms_thresh`** — soglia NMS sulle proposte.  
  *Esempi*: `0.7`⚡; `0.5` più severa (meno overlap).
- **`rpn.score_thresh`** — confidenza minima proposte (di rado si tocca).  
  *Esempi*: `0.0`⚡; `0.05` per filtrare rumore estremo.

### 4) RoI (RoIAlign + Box/Mask heads)
- **`roi_heads.batch_size_per_image`** — campioni/immagine per la box head.  
  *Esempi*: `512`⚡; `256` per ridurre memoria; `1024` per dataset piccoli.
- **`roi_heads.positive_fraction`** — quota positivi nel campionamento box.  
  *Esempi*: `0.25`⚡; `0.5` se hai molti positivi.
- **`box_fg_iou_thresh` / `box_bg_iou_thresh`** — IoU per match GT ↔ RoI.  
  *Esempi*: `0.5 / 0.5`⚡; `0.6 / 0.4` più esigente.
- **`roi_heads.score_thresh`** — confidenza minima in **inference**.  
  *Esempi*: `0.05`⚡ (recall alto), `0.3` (più precisione).
- **`roi_heads.nms_thresh`** — NMS sulle detezioni finali.  
  *Esempi*: `0.5`⚡; `0.3` (più severa), `0.7` (più permissiva).
- **`roi_heads.detections_per_img`** — max detezioni/immagine.  
  *Esempi*: `100`⚡; `20` per velocità/scene “pulite”.
- **`MaskRCNNPredictor.hidden_layer`** — canali della mask head.  
  *Esempi*: `256`⚡; `128` più leggero; `512` più capace.
- **`mask_roi_pool.output_size`** — risoluzione feature per maschere.  
  *Esempi*: `14`⚡; `28` per maschere più dettagliate (costo ↑).
- **Soglia binarizzazione maschera (post-proc)** — nel codice tipicamente `> 0.5`.  
  *Esempi*: `0.5`⚡; `0.3` per oggetti frastagliati; `0.7` per bordi netti.

### 5) Ottimizzazione (loop di training)
- **`optimizer`** — tipo e iperparametri.  
  *Esempi*: `SGD(lr=0.005, momentum=0.9, wd=5e-4)`⚡; `AdamW(lr=1e-4, wd=1e-4)`.
- **Scheduler LR** — come cambia il learning rate nel tempo.  
  *Esempi*: `MultiStepLR(milestones=[8,11], gamma=0.1)`; `CosineAnnealingLR(T_max=20)`.
- **`num_epochs` / `batch_size`** — durata training / compromesso tempo-memoria.  
  *Esempi*: `epochs=5`⚡ (quick fit), `epochs=20` con più dati. `batch_size=2`⚡; `4` se la GPU regge.

---

## Suggerimenti pratici
- Oggetti **molto piccoli** → riduci `min_size`, aumenta anchor piccole (`sizes=(8,16,32,...)`), abbassa `score_thresh` in inference.
- **Poche GPU/memoria** → `batch_size`↓, `min_size`↓, `post_nms_top_n_*`↓, `detections_per_img`↓.
- **Tante false positive** → alza `roi_heads.score_thresh` o abbassa `nms_thresh` (più severa).

