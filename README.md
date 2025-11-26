# OBJECT DETECTION AND TRACKING

Pada project berupa deteksi object yaitu tracking manusia/person.

---

## 🌟 Overview
Repo ini melakukan training dan testing detection menggunakan model YOLO dengan beberapa versi dan Faster R-CNN. 
| Model | Status | Catatan |
|-------|--------|---------|
| YOLOv8n | ✔️ Trained & Evaluated | Performa paling stabil |
| YOLOv8m | ✔️ Trained |
| YOLOv9s | ✔️ Trained |
| Faster R-CNN | ✔️ Baseline Model |

---
## 📊 Dataset
Dataset yang digunakan dari Roboflow open repo dan repo pribadi
| Dataset | Source |
|--------|--------|
| Public Dataset | 🔗 https://universe.roboflow.com/leo-ueno/people-detection-o4rdr/dataset/8 |
| Private Dataset | 🔗 https://app.roboflow.com/person-tracking-undvw |

Dataset berisi label **`person`** untuk object detection dan tracking.

---
## 📂 Repository Structure
.
├── Images/ # Image/video testing samples
├── streamlit/
│ ├── main.py # Streamlit UI for model testing
│ ├── yolo_utils/ # Utilities supporting YOLO inference
│ └── faster_rcnn_utils/ # Utilities supporting Faster R-CNN inference
├── downloaded_images.py # Automatic downloader from Unsplash
├── requirements.txt
├── test.ipynb # Testing and experiment notebook
└── training_yolov9.ipynb # Training documentation notebook

---
## 🔧 How to Run

### 1️⃣ Install Requirements and Run Streamlit
```sh
pip install -r requirements.txt
streamlit run streamlit/main.py
```

---
## Workflow model
https://github.com/MuhArifBudiman/Object-Detection-Tracking/blob/main/assets/flow.jpg

---
## Result
| Model        | Speed     | Accuracy               | Notes                             |
| ------------ | --------- | ---------------------- | --------------------------------- |
| YOLOv8n      | ⚡ Fast    | ⭐⭐ Good                | Best realtime + stable inference  |
| YOLOv8m      | ⚡⚡ Medium | ⭐⭐⭐ Higher capacity    | Slightly heavier                  |
| YOLOv9s      | ⚡⚡        | ⭐⭐⭐ Experimental       | Improved detections in edge cases |
| Faster R-CNN | ❌ Slow    | ⭐⭐⭐⭐ Accurate baseline | Not ideal for realtime            |


