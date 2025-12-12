import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from filterpy.kalman import KalmanFilter


st.set_page_config(page_title="KNN + Kalman Object Detection", layout="wide")

st.title("Object Detection")

# Sidebar controls for parameters
st.sidebar.header("Detection & Model Settings")
upload = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"]) 
n_neighbors = st.sidebar.slider("K (neighbors) for KNN", 1, 15, 3)
n_clusters = st.sidebar.slider("Clusters for initial KMeans", 1, 8, 3)
min_area = st.sidebar.slider("Min contour area", 50, 5000, 400)
bg_var_threshold = st.sidebar.slider("Background Subtractor varThreshold", 5, 100, 25)
use_kalman = st.sidebar.checkbox("Enable Kalman Filter Tracking", value=True)
bbox_color = st.sidebar.color_picker("Bounding box color", value="#00FF00")
bbox_thickness = st.sidebar.slider("Bounding box thickness", 1, 6, 2)
collect_frames = st.sidebar.slider("Frames to collect for clustering", 5, 120, 30)
fps_limit = st.sidebar.slider("Max FPS (processing)", 1, 30, 15)

st.sidebar.markdown("---")
play_button = st.sidebar.button("Play / Start Processing")
stop_button = st.sidebar.button("Stop")


def create_kalman():
    # 4 state: x, y, vx, vy ; 2 measurements: x, y
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.R *= 10.0
    kf.P *= 1000.0
    kf.Q = np.eye(4) * 0.01
    return kf


def extract_features(cnt, frame_shape):
    # Features: centroid x,y normalized, area, aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    cx = x + w / 2.0
    cy = y + h / 2.0
    fw, fh = frame_shape[1], frame_shape[0]
    return [cx / fw, cy / fh, area / (fw * fh), float(w) / float(h + 1e-6)]


def process_frame(frame, bg_subtractor, knn_model, labeler, trackers, settings):
    # Preprocessing: grayscale, blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Background subtraction to get motion mask
    fgmask = bg_subtractor.apply(blur)
    _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
    # Morphological ops to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < settings['min_area']:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        feat = extract_features(cnt, frame.shape)
        detections.append({'bbox': (x, y, w, h), 'feature': feat, 'centroid': (int(x + w/2), int(y + h/2)), 'area': area})

    # If we have a trained KNN, classify features
    for det in detections:
        if knn_model is not None and len(detections) > 0:
            pred = knn_model.predict([det['feature']])[0]
        else:
            pred = -1
        det['label'] = pred

    # Update trackers (Kalman) per label
    for det in detections:
        lbl = det['label']
        cx, cy = det['centroid']
        if settings['use_kalman']:
            if lbl not in trackers:
                kf = create_kalman()
                kf.x = np.array([cx, cy, 0, 0], dtype=float)
                trackers[lbl] = {'kf': kf, 'last_seen': time.time()}
            else:
                trackers[lbl]['kf'].predict()
                trackers[lbl]['kf'].update(np.array([cx, cy]))
                trackers[lbl]['last_seen'] = time.time()
            # replace centroid by Kalman estimate for smoother drawing
            kf = trackers[lbl]['kf']
            det['kf_centroid'] = (int(kf.x[0]), int(kf.x[1]))
        else:
            # no Kalman: just use current centroid
            det['kf_centroid'] = det['centroid']

    # Remove stale trackers
    stale = []
    for lbl, info in trackers.items():
        if time.time() - info['last_seen'] > 1.5:
            stale.append(lbl)
    for lbl in stale:
        del trackers[lbl]

    # Visualization: draw bboxes and labels
    out = frame.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        cx, cy = det['kf_centroid']
        label = det.get('label', -1)
        color = settings['color']
        # convert hex color to BGR tuple
        hexc = color.lstrip('#')
        bc = tuple(int(hexc[i:i+2], 16) for i in (0, 2, 4))
        bc = (bc[2], bc[1], bc[0])
        cv2.rectangle(out, (x, y), (x+w, y+h), bc, settings['thickness'])
        cv2.circle(out, (cx, cy), 3, bc, -1)
        cv2.putText(out, f"ID:{label}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bc, 1)

    return out, thresh


def main():
    st.sidebar.markdown("Upload a video and press Play to start processing.")

    # Placeholders for side-by-side display
    col1, col2 = st.columns(2)
    orig_slot = col1.empty()
    proc_slot = col2.empty()

    if upload is None:
        st.info("Please upload a video file from the sidebar to begin.")
        return

    # Save uploaded to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Unable to open uploaded video.")
        return

    # Prepare background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=bg_var_threshold, detectShadows=False)

    # Data collection buffers for initial clustering
    collected_features = []
    collected_labels = []
    knn_model = None
    labeler = None
    trackers = {}

    # Session control
    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if play_button:
        st.session_state.playing = True
    if stop_button:
        st.session_state.playing = False

    # Settings pack
    settings = {'min_area': min_area, 'use_kalman': use_kalman, 'color': bbox_color, 'thickness': bbox_thickness}

    frame_time = 1.0 / max(1, fps_limit)

    # Frame processing loop
    frame_idx = 0
    while cap.isOpened() and st.session_state.playing:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Downscale large frames for speed
        h, w = frame.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h,w))
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        # Collect features during warm-up for clustering
        gray_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_tmp = cv2.GaussianBlur(gray_tmp, (5,5), 0)
        fgmask_tmp = bg_subtractor.apply(blur_tmp)
        _, thresh_tmp = cv2.threshold(fgmask_tmp, 244, 255, cv2.THRESH_BINARY)
        contours_tmp, _ = cv2.findContours(thresh_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_tmp:
            if cv2.contourArea(cnt) < settings['min_area']:
                continue
            feat = extract_features(cnt, frame.shape)
            collected_features.append(feat)

        # After collecting enough frames, build cluster labels and KNN
        if frame_idx == collect_frames and len(collected_features) > 0:
            # Use KMeans to create pseudo-labels for KNN training
            try:
                kmeans = KMeans(n_clusters=min(n_clusters, max(1, len(collected_features))), random_state=0).fit(collected_features)
                labels = kmeans.labels_
                knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn_model.fit(collected_features, labels)
                labeler = kmeans
            except Exception:
                knn_model = None

        # If knn exists, we will use it. Process and show
        out_frame, mask = process_frame(frame, bg_subtractor, knn_model, labeler, trackers, settings)

        # Convert frames for Streamlit
        orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)

        # compute a reasonable display width to replace deprecated use_column_width
        display_w = min(800, frame.shape[1])
        orig_slot.image(orig_rgb, caption="Original", width=display_w)
        proc_slot.image(out_rgb, caption="Detection Output", width=display_w)

        time.sleep(frame_time)

    cap.release()

    st.sidebar.markdown("Processing stopped.")


if __name__ == '__main__':
    main()
