# KNN + Kalman Object Detection (Streamlit)

This Streamlit app performs simple object detection and tracking without deep learning. It uses background subtraction to find moving objects, extracts simple features, clusters initial observations with KMeans to create pseudo-labels, trains a KNN classifier, and optionally uses a Kalman filter to smooth tracking.

Features:
- Upload a video and run processing in the browser.
- Adjustable K for KNN, number of clusters for initial KMeans, thresholds, and Kalman toggle.
- Side-by-side display of original and processed frames.

Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Notes:
- No TensorFlow / PyTorch or other deep learning libraries are used.
- The app collects a number of frames (set in the sidebar) to form initial clusters; this warm-up phase helps the KNN to get pseudo-labels.
- Kalman tracking is per-cluster label; turn it off to see raw detections.
