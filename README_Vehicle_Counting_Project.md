# Vehicle Detection, Tracking & Counting System
### YOLO + Deep OC-SORT + ROI Filtering + Line-Crossing + Streamlit

A complete computer-vision pipeline for detecting, tracking, classifying, and counting vehicles in traffic videos using a custom-trained YOLO model and Deep OC-SORT. The project includes Jupyter-based experimentation, tracking and counting logic, class-level majority voting, a Streamlit web application, GitHub version control with Git LFS, and deployment on Streamlit Community Cloud.

---

## 1. Project Objective

The goal of this project is to build an end-to-end traffic video analytics system that can:

- Detect vehicles in a traffic video using a custom-trained YOLO model
- Track each detected vehicle with a persistent ID
- Restrict analysis to a road Region of Interest (ROI)
- Count vehicles when they cross a virtual counting line
- Avoid double counting by using unique tracking IDs
- Improve class prediction using track-level majority voting
- Display the final counts by vehicle category
- Provide a Streamlit interface for uploading and processing videos
- Deploy the complete application online

---

## 2. Final Pipeline

```text
Traffic Video
     ↓
Custom YOLO Model (best.pt)
     ↓
Vehicle Detection
     ↓
Road ROI Filtering
     ↓
Deep OC-SORT Tracking
     ↓
Persistent Track IDs
     ↓
Bottom-Center Vehicle Point
     ↓
Virtual Line-Crossing Detection
     ↓
Unique Vehicle Counting
     ↓
Track-Level Majority Voting
     ↓
Class-Wise Vehicle Count
     ↓
Streamlit Web Application
```

---

## 3. Vehicle Classes

The trained YOLO model contains the following classes:

```text
0 → MTW
1 → CAR
2 → 3W
3 → LCV
4 → HCV
```

Where:

- **MTW** = Motorized Two-Wheeler
- **CAR** = Passenger Car
- **3W** = Three-Wheeler
- **LCV** = Light Commercial Vehicle
- **HCV** = Heavy Commercial Vehicle

---

## 4. Project Folder Structure

```text
Vehicle_Counting_App/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
│
├── models/
│   └── best.pt
│
├── Vehicle_Counting.ipynb
│
└── test_videos/              # optional local-only folder
```

The trained YOLO model is stored at:

```text
models/best.pt
```

---

# 5. Development Workflow

## Step 1 — Create the project folder

```text
Vehicle_Counting_App/
```

Create a `models` folder and place the trained YOLO weight inside:

```text
Vehicle_Counting_App/models/best.pt
```

Place a test traffic video in the project folder, for example:

```text
test2.mp4
```

---

## Step 2 — Install required Python packages

In Jupyter Notebook:

```python
%pip install -U ultralytics opencv-python numpy pandas streamlit dill lap
```

For deployment, the final `requirements.txt` is:

```text
streamlit==1.62.0
ultralytics
opencv-python-headless
numpy
dill
lap>=0.5.12
```

`dill` was required to load the serialized YOLO model on Streamlit Cloud.

`lap` is required by the Ultralytics tracking utilities used by Deep OC-SORT.

---

## Step 3 — Load the YOLO model

```python
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"

model = YOLO(MODEL_PATH)

print(model.names)
```

Expected classes:

```python
{
    0: "MTW",
    1: "CAR",
    2: "3W",
    3: "LCV",
    4: "HCV"
}
```

---

## Step 4 — Check the input video

```python
import cv2

VIDEO_PATH = "test2.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Width:", width)
print("Height:", height)
print("FPS:", fps)
print("Frames:", total_frames)

cap.release()
```

For one of the test videos:

```text
Width: 720
Height: 1280
FPS: ≈30
Frames: 1207
```

---

# 6. YOLO Detection Testing

Before tracking, the detector was tested on individual frames.

```python
results = model.predict(
    frame,
    conf=0.25,
    iou=0.50,
    imgsz=1280,
    verbose=False
)
```

### Confidence Threshold

The confidence threshold is **not the model accuracy**.

For example:

```text
CAR 0.91  → kept
MTW 0.76  → kept
CAR 0.27  → kept
HCV 0.18  → rejected
```

when:

```text
confidence threshold = 0.25
```

A lower threshold detects more vehicles but can create more false positives.

A higher threshold removes weak detections but may miss small or distant vehicles.

For this project, a useful starting point was:

```text
Confidence Threshold = 0.25
Image Size = 1280
```

---

# 7. False-Positive Filtering with Road ROI

The detector initially produced some false detections on background structures.

To reduce them, a polygon-based Region of Interest was created for the road.

```python
import numpy as np

road_roi = np.array([
    [
        (int(width * 0.25), int(height * 0.20)),
        (int(width * 0.78), int(height * 0.20)),
        (int(width * 0.93), int(height * 0.98)),
        (int(width * 0.05), int(height * 0.98))
    ]
], dtype=np.int32)
```

The bottom-center point of every detected object is checked:

```python
inside_road = (
    cv2.pointPolygonTest(
        road_roi[0],
        (point_x, point_y),
        False
    ) >= 0
)
```

Only objects inside the road ROI are retained.

---

## Large False-Detection Filter

Very large false bounding boxes are rejected using area ratio:

```python
box_area = (x2 - x1) * (y2 - y1)

frame_area = width * height

area_ratio = box_area / frame_area

too_large = area_ratio > 0.12
```

Then:

```python
if not inside_road or too_large:
    continue
```

This removes detections such as a bridge or large background region incorrectly classified as a vehicle.

---

# 8. Deep OC-SORT Tracking

The final tracker used in this project is **Deep OC-SORT**.

```python
results = model.track(
    frame,
    persist=True,
    tracker="deepocsort.yaml",
    conf=0.25,
    imgsz=1280,
    iou=0.50,
    verbose=False
)
```

### Why tracking is needed

Without tracking:

```text
Frame 1 → Car detected
Frame 2 → Same car detected
Frame 3 → Same car detected
...
```

The same vehicle could be counted repeatedly.

With tracking:

```text
Frame 1 → CAR ID 12
Frame 2 → CAR ID 12
Frame 3 → CAR ID 12
```

The same vehicle maintains one ID.

---

# 9. ByteTrack vs Deep OC-SORT

Two common tracking options are:

### ByteTrack

Advantages:

- Faster
- Lower computational cost
- Simple
- Strong baseline

### Deep OC-SORT

Advantages:

- Better identity continuity
- Better handling of occlusion
- More suitable for crowded traffic
- Observation-centric tracking
- Optional appearance/ReID support

For this project, **Deep OC-SORT** was selected because vehicle counting depends heavily on stable IDs.

---

# 10. Bottom-Center Tracking Point

For line-crossing, the center of the bounding box was replaced with the bottom-center point.

```python
point_x = int((x1 + x2) / 2)
point_y = int(y2)
```

This is preferred because the bottom of a vehicle bounding box approximately represents the vehicle's position on the road.

---

# 11. Virtual Counting Line

A horizontal counting line is defined as a fraction of the video height.

```python
line_y = int(height * 0.60)

line_x1 = int(width * 0.15)
line_x2 = int(width * 0.85)
```

The application checks whether the vehicle moves from above the line to below it.

```python
crossed_down = (
    previous_y < line_y
    and
    point_y >= line_y
)
```

To prevent double counting:

```python
counted_ids = set()
```

and:

```python
if track_id not in counted_ids:
    counted_ids.add(track_id)
```

Each vehicle ID can therefore be counted only once.

---

# 12. Track-Level Majority Voting

A major issue in video detection is that the same tracked vehicle may receive different class labels in different frames.

Example:

```text
ID 25:
Frame 1 → MTW
Frame 2 → MTW
Frame 3 → CAR
Frame 4 → MTW
Frame 5 → MTW
```

Instead of trusting a single frame, all class predictions for the track are stored:

```python
track_class_votes[track_id].append(class_name)
```

Then the final class is selected using majority voting:

```python
from collections import Counter

majority_class = Counter(votes).most_common(1)[0][0]
```

This significantly improved class-level counting stability.

---

# 13. Example Final Result

For one test video, the final track-level majority-voting result was:

```text
Total Vehicles: 85

CAR : 25
MTW : 48
3W  : 9
HCV : 2
LCV : 1
```

Verification:

```text
25 + 48 + 9 + 2 + 1 = 85
```

Earlier, frame-level classification produced too many HCV predictions. Track-level majority voting reduced those temporary misclassifications substantially.

---

# 14. Streamlit Application

The final system was converted into a Streamlit application.

The interface allows the user to:

- Upload traffic video
- Select confidence threshold
- Select YOLO inference image size
- Select counting-line position
- Run YOLO + Deep OC-SORT
- View processing progress
- View total vehicle count
- View class-wise counts
- Download the processed video

---

## Run Streamlit Locally

From the project folder:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

Recommended initial settings:

```text
Confidence Threshold = 0.25
YOLO Image Size = 1280
Counting Line Position = 0.60
```

---

# 15. GitHub Setup

The trained `best.pt` file is approximately 137 MB, which is larger than GitHub's normal file-size limit.

Therefore **Git LFS** is used.

Install and initialize Git LFS:

```bash
git lfs install
```

Track YOLO weight files:

```bash
git lfs track "*.pt"
```

This creates:

```text
.gitattributes
```

Verify:

```bash
git lfs ls-files
```

Expected result:

```text
models/best.pt
```

---

## Git Repository Setup

```bash
git init
git branch -M main
```

Add the important files:

```bash
git add app.py
git add requirements.txt
git add .gitignore
git add .gitattributes
git add models/best.pt
```

Commit:

```bash
git commit -m "Add YOLO Deep OC-SORT vehicle counting app"
```

Connect GitHub:

```bash
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
```

Push:

```bash
git push -u origin main
```

If the remote repository already contains a README:

```bash
git pull origin main --allow-unrelated-histories --no-rebase
```

Then:

```bash
git push -u origin main
```

---

# 16. `.gitignore`

A useful `.gitignore` for this project:

```text
# Jupyter
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc

# Test videos
test.mp4
test2.mp4
test3.mp4

# Generated videos
*output*.mp4
counted_output*.mp4

# Generated test images
*.jpg
*.jpeg
*.png

# Temporary files
*.tmp
```

This keeps large test videos and temporary debugging images out of GitHub.

---

# 17. Streamlit Cloud Deployment

The project was deployed using Streamlit Community Cloud.

Deployment configuration:

```text
Repository:
<GitHub username>/<repository name>

Branch:
main

Main file:
app.py
```

Streamlit then builds the application automatically from:

```text
requirements.txt
```

---

# 18. Deployment Errors and Fixes

## Error 1 — `ModuleNotFoundError: No module named 'dill'`

The serialized YOLO model required `dill`.

Fix:

```text
dill
```

was added to `requirements.txt`.

---

## Error 2 — `ModuleNotFoundError: No module named 'lap'`

Ultralytics tracking requires the `lap` package.

Fix:

```text
lap>=0.5.12
```

was added to `requirements.txt`.

---

## Final `requirements.txt`

```text
streamlit==1.62.0
ultralytics
opencv-python-headless
numpy
dill
lap>=0.5.12
```

---

# 19. Updating the Deployed App

Whenever `app.py` or dependencies are changed:

```bash
git add app.py requirements.txt
git commit -m "Update vehicle counting app"
git pull --rebase origin main
git push
```

Streamlit Community Cloud detects the GitHub update and redeploys the application.

---

# 20. Current Limitations

The current system still has some limitations:

- The ROI is camera-specific
- The counting line is manually positioned
- Very small distant vehicles may be missed
- YOLO may occasionally misclassify similar vehicle types
- Tracking can still experience ID switches under severe occlusion
- Performance depends on video resolution and hardware
- High inference image sizes improve small-object detection but reduce processing speed

---

# 21. Future Improvements

Possible improvements include:

- Automatic road ROI selection
- User-drawn ROI and counting line in Streamlit
- Two-way directional counting
- Separate lane-wise counts
- Adaptive confidence thresholding
- Deep OC-SORT with ReID enabled
- ByteTrack vs Deep OC-SORT benchmark
- ID-switch analysis
- Manual ground-truth counting and counting-error evaluation
- Precision, Recall, F1, mAP, IDF1 and HOTA evaluation
- Real-time CCTV/RTSP stream support
- Database storage for historical traffic analytics
- Interactive charts and traffic dashboards
- Vehicle speed estimation

---

# 22. Recommended Tracker Comparison Experiment

For a stronger project, compare:

```text
YOLO + ByteTrack
vs
YOLO + Deep OC-SORT
```

Suggested metrics:

- Manual vehicle count
- Predicted vehicle count
- Absolute counting error
- Percentage counting error
- ID switches
- Tracking FPS
- Processing time
- IDF1
- HOTA

Example reporting format:

| Tracker | Manual Count | Predicted Count | Count Error | FPS |
|---|---:|---:|---:|---:|
| ByteTrack | 85 | 89 | 4 | Higher |
| Deep OC-SORT | 85 | 86 | 1 | Lower |

This makes the project more rigorous than simply selecting one tracker.

---

# 23. Technologies Used

- Python
- YOLO / Ultralytics
- Deep OC-SORT
- OpenCV
- NumPy
- Streamlit
- Git
- GitHub
- Git LFS
- Streamlit Community Cloud
- Jupyter Notebook

---

# 24. Summary

This project demonstrates a complete computer-vision deployment workflow:

```text
Model Training
      ↓
YOLO Detection
      ↓
False-Positive Filtering
      ↓
Deep OC-SORT Tracking
      ↓
Track ID Management
      ↓
Line-Crossing Counting
      ↓
Track-Level Majority Voting
      ↓
Streamlit Interface
      ↓
GitHub + Git LFS
      ↓
Cloud Deployment
```

The final application provides a practical traffic-analysis interface where users can upload a traffic video and obtain unique vehicle counts together with class-wise statistics.

---

## Author

Built as an end-to-end computer-vision project for vehicle detection, multi-object tracking, traffic counting, and web deployment.
