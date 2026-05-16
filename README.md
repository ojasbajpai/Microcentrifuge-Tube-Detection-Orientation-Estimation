# Microcentrifuge Tube Detection & Orientation Estimation

A classical computer vision system that detects microcentrifuge tube lid **positions and orientations** from overhead RGB images — no deep learning, no training data, pure OpenCV.

---

## Final Results

| Metric | Value |
|---|---|
| Total GT Tubes | 371 |
| True Positives (TP) | 367 |
| False Positives (FP) | 6 |
| False Negatives (FN) | 4 |
| **Precision** | **0.984** |
| **Recall** | **0.989** |
| **F1 Score** | **0.987** |
| Mean Center Error | 3.86 px |
| Median Center Error | 3.58 px |
| Mean Angle Error | 71.04° |
| Median Angle Error | 22.94° |

---

## Dataset

70 overhead RGB images (640×480) of microcentrifuge tubes in a dark foam tray, on various backgrounds (wood desk, white wall, black surface, concrete floor, cluttered workbench with tools).

Each image has 3–6 tubes. Annotations are in `annotations.csv`:

| Column | Description |
|---|---|
| `image` | filename |
| `center_x`, `center_y` | lid center in pixels |
| `bbox_x/y/w/h` | bounding box |
| `angle_deg` | rotation [0°, 360°), CCW from +x, defined as joint→tab direction |

---

## Pipeline

```
Input Image
    │
    ▼
find_tray()         ← HoughCircles on dark grid holes → tight search region
    │
    ▼
segment_lids()      ← Difference of Gaussians → lighting-invariant blob detection
    │
    ▼
detect_tubes()      ← Moments for center, PCA + brightness sampling for angle
    │
    ▼
evaluate()          ← Greedy nearest-neighbour matching vs ground truth
```

---

## Design Decisions & How We Got There

### Step 1 — Finding the Tray (`find_tray`)

**Challenge:** Backgrounds vary wildly — wood, white, black, concrete, tools. A fixed brightness or colour threshold that works on one background fails on another.

**What we tried and why it failed:**

- *Global brightness threshold* — the wood desk is similar brightness to the tray on some images. Failed on bright backgrounds.
- *HSV colour thresholding* — lids and backgrounds overlap in colour space across different images.
- *"Largest dark rectangle"* — a phone or laptop is often larger and darker than the tray.
- *Blob-count scoring* — picked the yellow equipment rack (which has more circles) over the tray.

**What works — hole-cluster detection:**

The foam tray always has a **regular 4×4 grid of dark circular holes**. We use `cv2.HoughCircles` to find all circles in the image, then score each dense cluster by three factors:

```
score = num_neighbors × darkness_factor × neutrality²

darkness_factor  = 1 - (mean_brightness / 255)   # tray is dark foam
neutrality       = 1 - (mean_saturation / 255)    # foam has no colour
```

This discriminates correctly:
- Yellow rack: many circles but high saturation → low score
- Phone: dark but no circles inside it → no cluster
- Foam tray: dark + neutral + dense grid → highest score

**Additional fix — verify real holes:** Metal objects create Hough circles via reflection. Each circle in the winning cluster is verified: a real hole has `mean_interior < mean_surrounding − 5`. Reflections are bright inside and fail this check.

**Padding:** 18px fixed padding around the verified hole bounding box. Tight enough to exclude objects below the tray, generous enough to include tabs that protrude from edge holes.

---

### Step 2 — Lid Segmentation (`segment_lids`)

**Challenge:** Tube lids are translucent white/grey — visually similar to many backgrounds. Their absolute brightness varies enormously by image.

**What we tried and why it failed:**

- *Global brightness threshold* — works on dark backgrounds, fails on white walls.
- *Adaptive thresholding* — the tray border creates a huge local brightness step, the boundary pixels dominate and raise the threshold until real lids disappear.
- *Otsu inside ROI* — the pixel histogram inside the tray has no clean valley between foam surface and lids. Otsu fires too early (at foam pixels).
- *75th percentile threshold* — depends on how many lids are present (3 vs 6 tubes = different distributions). Not stable.

**What works — Difference of Gaussians (DoG):**

```python
DoG = GaussianBlur(σ=3) − GaussianBlur(σ=18)
```

This measures *"is this pixel brighter than its ~18px neighbourhood?"*. Tube lids always sit brighter than the dark holes immediately around them — regardless of the absolute brightness of the entire image. The DoG response is **lighting-invariant by construction**.

Key implementation details:
- **Border ring suppression:** Erode the region mask by 18px before computing threshold statistics. The tray edge has a massive DoG response (dark foam → bright background transition). If we include boundary pixels when computing the adaptive threshold, the threshold inflates and real lids disappear. Excluding the 18px border ring fixes this.
- **Adaptive threshold from inner pixels:** `threshold = mean(DoG_inner) + 0.8 × std(DoG_inner)`. Automatically calibrates to each image's contrast.
- **Flood fill:** Translucent lids have darker centres (you can see through them to the dark hole beneath). This creates donut-shaped blobs. Flood fill from image corner + invert closes these holes.
- **Contour filtering:** area 300–6000 px², circularity > 0.28, must not touch image boundary.

---

### Step 3 — Center & Orientation (`detect_tubes`)

**Center:** Image moments (`cv2.moments`) give the blob centroid. Accurate to ~4px across all backgrounds.

**Orientation — the hard problem:**

The annotation defines angle as the **joint-to-tab direction** (full 360°). The tab is a small rectangular protrusion on one side; the joint (hinge) is on the other side of the round cap.

**What we tried first — farthest contour point:**
```python
farthest_point = contour[argmax(distance_from_centroid)]
angle = atan2(farthest_y - cy, farthest_x - cx)
```
This gave **mean angle error = 123°, median = 162°**. The tab *is* usually the farthest point, but sometimes the joint side of the contour is slightly farther (especially when the segmentation blob is imperfect). This causes systematic 180° flips.

**What works — PCA + brightness sampling beyond the blob:**

1. `cv2.PCACompute` on blob contour pixels finds the main axis robustly. This is stable even on rough blobs.
2. PCA gives two candidate directions (±). To resolve the 180° ambiguity, we sample pixel brightness **just beyond the blob boundary** in each direction:
   - Sample at 0.8×–1.3× the blob extent → "near" brightness
   - Sample at 1.0×–1.8× the blob extent → "far" brightness  
   - `score = near_brightness + 0.5 × far_brightness`

The **tab protrudes further** from the centroid than the joint. So in the tab direction, brightness stays high at larger distances (we're still sampling the tab itself). In the joint direction, we quickly hit dark tray.

This reduced mean angle error from **123° → 71°** and median from **162° → 23°**.

The remaining ~85 flipped tubes (out of 367) are cases where the segmentation blob doesn't capture the tab at all — the tab is cut off by the tray boundary, or merged with the border. Without the tab pixels in the blob, no downstream method can recover the direction.

---

### Step 4 — Evaluation (`evaluate`)

Greedy nearest-neighbour matching with 30px distance threshold:
- Each GT tube is matched to the closest unmatched prediction within 30px → **True Positive**
- Unmatched predictions → **False Positives**
- Unmatched GT → **False Negatives**
- Angular error: `min(|pred − gt|, 360 − |pred − gt|)` — shortest arc on 360° circle

Images with wrong detection counts still contribute their correctly-matched pairs to center/angle error statistics.

---

## Usage

### Requirements
```bash
pip install opencv-python numpy pandas matplotlib
```

### Run in Google Colab
1. Upload the dataset folder to Google Drive
2. Open `microcentrifuge_tube_detection.ipynb` in Colab
3. Change `DATASET_PATH` in Cell 2 to your folder path
4. Runtime → Run all

### Expected folder structure
```
YOUR_FOLDER/
├── images/
│   ├── 2659ffa5-color.png
│   ├── ...
└── annotations.csv
```

---

## Key CV Concepts

| Concept | Where Used |
|---|---|
| Hough Circle Transform | Detecting tray holes |
| HSV saturation scoring | Rejecting coloured non-tray objects |
| Difference of Gaussians | Lighting-invariant lid detection |
| Morphological operations (open/close) | Noise removal and hole filling |
| Flood fill | Closing donut-shaped blob centres |
| Image moments | Computing blob centroids |
| PCA on contour pixels | Finding tube main axis |
| Brightness profile sampling | Resolving 180° tab/joint ambiguity |
| Greedy bipartite matching | Prediction ↔ ground truth association |
