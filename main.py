# ===================================
# Object Detection Application
# Using KNN Classification + Kalman Filter Tracking
# ===================================

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from filterpy.kalman import KalmanFilter

# Configure Streamlit page settings
st.set_page_config(page_title="Object Detection", layout="wide")

st.title("Object Detection")

# ===================================
# Sidebar Controls - User Parameters
# ===================================
st.sidebar.header("Detection & Model Settings")

# Video upload widget
upload = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"]) 

# Machine Learning parameters
n_neighbors = st.sidebar.slider("K (neighbors) for KNN", 1, 15, 3)  # KNN classifier parameter
n_clusters = st.sidebar.slider("Clusters for initial KMeans", 1, 8, 3)  # Number of object types

# Detection parameters
min_area = st.sidebar.slider("Min contour area", 50, 5000, 400)  # Filter small detections
bg_var_threshold = st.sidebar.slider("Background Subtractor varThreshold", 5, 100, 25)  # Sensitivity

# Tracking and visualization options
use_kalman = st.sidebar.checkbox("Enable Kalman Filter Tracking", value=True)  # Smooth tracking
bbox_color = st.sidebar.color_picker("Bounding box color", value="#00FF00")  # Green default
bbox_thickness = st.sidebar.slider("Bounding box thickness", 1, 6, 2)

# Processing parameters
collect_frames = st.sidebar.slider("Frames to collect for clustering", 5, 120, 30)  # Training data
fps_limit = st.sidebar.slider("Max FPS (processing)", 1, 30, 15)  # Control processing speed

# Control buttons
st.sidebar.markdown("---")
play_button = st.sidebar.button("Play / Start Processing")
stop_button = st.sidebar.button("Stop")


# ===================================
# Kalman Filter Setup
# ===================================
def create_kalman():
    """
    Create a Kalman Filter for smooth object tracking
    State vector: [x, y, vx, vy] - position and velocity
    Measurement: [x, y] - observed position
    """
    # 4 state variables: x, y, vx, vy ; 2 measurements: x, y
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    # State transition matrix (F): predicts next state from current state
    kf.F = np.array([[1, 0, 1, 0],  # x = x + vx
                     [0, 1, 0, 1],  # y = y + vy
                     [0, 0, 1, 0],  # vx = vx (constant velocity)
                     [0, 0, 0, 1]]) # vy = vy
    
    # Measurement matrix (H): extracts position from state
    kf.H = np.array([[1, 0, 0, 0],  # measure x
                     [0, 1, 0, 0]]) # measure y
    
    # Noise covariance matrices
    kf.R *= 10.0      # Measurement noise (observation uncertainty)
    kf.P *= 1000.0    # Initial state covariance (initial uncertainty)
    kf.Q = np.eye(4) * 0.01  # Process noise (model uncertainty)
    return kf


# ===================================
# Feature Extraction
# ===================================
def extract_features(cnt, frame_shape):
    """
    Extract features from a detected contour
    Returns: [normalized_x, normalized_y, normalized_area, aspect_ratio]
    """
    # Get bounding rectangle and calculate features
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    cx = x + w / 2.0
    cy = y + h / 2.0
    fw, fh = frame_shape[1], frame_shape[0]
    return [cx / fw, cy / fh, area / (fw * fh), float(w) / float(h + 1e-6)]


# ===================================
# Frame Processing Pipeline
# ===================================
def process_frame(frame, bg_subtractor, knn_model, labeler, trackers, settings):
    """
    Main processing pipeline for each frame
    Steps: preprocessing → background subtraction → contour detection → 
           feature extraction → classification → tracking → visualization
    """
    # Step 1: Preprocessing - reduce noise and prepare for background subtraction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Gaussian blur to reduce noise

    # Step 2: Background subtraction to detect moving objects
    fgmask = bg_subtractor.apply(blur)
    _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)  # Binary threshold
    
    # Step 3: Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)   # Remove noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=2) # Fill gaps

    # Step 4: Find contours (object boundaries)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Extract features and create detection objects
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter out small contours (noise)
        if area < settings['min_area']:
            continue
        
        # Get bounding box and extract features
        x, y, w, h = cv2.boundingRect(cnt)
        feat = extract_features(cnt, frame.shape)
        
        # Store detection information
        detections.append({
            'bbox': (x, y, w, h), 
            'feature': feat, 
            'centroid': (int(x + w/2), int(y + h/2)), 
            'area': area
        })

    # Step 6: Classify detections using trained KNN model
    for det in detections:
        if knn_model is not None and len(detections) > 0:
            # Predict object class based on extracted features
            pred = knn_model.predict([det['feature']])[0]
        else:
            # Model not trained yet, use placeholder label
            pred = -1
        det['label'] = pred

    # Step 7: Update Kalman Filter trackers for smooth tracking
    for det in detections:
        lbl = det['label']
        cx, cy = det['centroid']
        
        if settings['use_kalman']:
            if lbl not in trackers:
                # Create new tracker for this object class
                kf = create_kalman()
                kf.x = np.array([cx, cy, 0, 0], dtype=float)  # Initialize with current position
                trackers[lbl] = {'kf': kf, 'last_seen': time.time()}
            else:
                # Update existing tracker
                trackers[lbl]['kf'].predict()  # Predict next position
                trackers[lbl]['kf'].update(np.array([cx, cy]))  # Correct with measurement
                trackers[lbl]['last_seen'] = time.time()
            
            # Use Kalman-smoothed position for display
            kf = trackers[lbl]['kf']
            det['kf_centroid'] = (int(kf.x[0]), int(kf.x[1]))
        else:
            # Kalman disabled: use raw detection centroid
            det['kf_centroid'] = det['centroid']

    # Step 8: Remove stale trackers (objects no longer visible)
    stale = []
    for lbl, info in trackers.items():
        if time.time() - info['last_seen'] > 1.5:  # 1.5 seconds timeout
            stale.append(lbl)
    for lbl in stale:
        del trackers[lbl]

    # Step 9: Visualization - draw bounding boxes, centroids, and labels
    out = frame.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        cx, cy = det['kf_centroid']  # Use Kalman-smoothed position
        label = det.get('label', -1)
        
        # Convert hex color to BGR tuple for OpenCV
        color = settings['color']
        hexc = color.lstrip('#')
        bc = tuple(int(hexc[i:i+2], 16) for i in (0, 2, 4))  # RGB
        bc = (bc[2], bc[1], bc[0])  # Convert to BGR
        
        # Draw bounding box
        cv2.rectangle(out, (x, y), (x+w, y+h), bc, settings['thickness'])
        # Draw centroid point
        cv2.circle(out, (cx, cy), 3, bc, -1)
        # Draw label
        cv2.putText(out, f"ID:{label}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bc, 1)

    return out, thresh


# ===================================
# Main Application
# ===================================
def main():
    """Main application logic for video processing and display"""
    st.sidebar.markdown("Upload a video and press Play to start processing.")

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    orig_slot = col1.empty()  # Placeholder for original video
    proc_slot = col2.empty()  # Placeholder for processed video

    # Check if video is uploaded
    if upload is None:
        st.info("Please upload a video file from the sidebar to begin.")
        return

    # Save uploaded file to temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload.read())
    tfile.flush()

    # Open video capture
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Unable to open uploaded video.")
        return

    # Initialize background subtractor (MOG2 algorithm)
    # Learns background model over time to detect moving foreground objects
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,                      # Number of frames for background model
        varThreshold=bg_var_threshold,    # Threshold for pixel classification
        detectShadows=False               # Disable shadow detection for speed
    )

    # Initialize model components
    collected_features = []  # Features collected during warm-up phase
    collected_labels = []    # Labels from KMeans clustering
    knn_model = None         # KNN classifier (trained after warm-up)
    labeler = None           # KMeans model for clustering
    trackers = {}            # Dictionary of Kalman filters per object class

    # Session state for play/stop control
    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if play_button:
        st.session_state.playing = True
    if stop_button:
        st.session_state.playing = False

    # Pack settings into dictionary for easy passing
    settings = {
        'min_area': min_area,           # Minimum area threshold
        'use_kalman': use_kalman,       # Enable Kalman filtering
        'color': bbox_color,            # Bounding box color
        'thickness': bbox_thickness     # Bounding box line thickness
    }

    # Calculate frame delay for FPS limiting
    frame_time = 1.0 / max(1, fps_limit)

    # ===================================
    # Main Video Processing Loop
    # ===================================
    frame_idx = 0
    while cap.isOpened() and st.session_state.playing:
        ret, frame = cap.read()
        if not ret:  # End of video
            break

        frame_idx += 1

        # Optimize performance: downscale large frames
        h, w = frame.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        # ===================================
        # Phase 1: Warm-up - Collect training data
        # ===================================
        # Collect features from first N frames for KMeans clustering
        gray_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_tmp = cv2.GaussianBlur(gray_tmp, (5,  5), 0)
        fgmask_tmp = bg_subtractor.apply(blur_tmp)
        _, thresh_tmp = cv2.threshold(fgmask_tmp, 244, 255, cv2.THRESH_BINARY)
        contours_tmp, _ = cv2.findContours(thresh_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract and store features from all detected objects
        for cnt in contours_tmp:
            if cv2.contourArea(cnt) < settings['min_area']:
                continue
            feat = extract_features(cnt, frame.shape)
            collected_features.append(feat)

        # ===================================
        # Phase 2: Train Models
        # ===================================
        # After collecting enough features, train KMeans and KNN
        if frame_idx == collect_frames and len(collected_features) > 0:
            # Two-stage learning:
            # 1. KMeans: Unsupervised clustering to discover object classes
            # 2. KNN: Supervised classifier trained on cluster labels
            try:
                # Ensure we have enough samples for training
                n_samples = len(collected_features)
                if n_samples < 2:
                    # Not enough samples to train, skip training
                    knn_model = None
                else:
                    # Step 1: Cluster features into object types
                    kmeans = KMeans(
                        n_clusters=min(n_clusters, max(1, n_samples)), 
                        random_state=0
                    ).fit(collected_features)
                    labels = kmeans.labels_
                    
                    # Step 2: Train KNN classifier with appropriate n_neighbors
                    # Ensure n_neighbors doesn't exceed number of samples
                    actual_neighbors = min(n_neighbors, n_samples)
                    knn_model = KNeighborsClassifier(n_neighbors=actual_neighbors)
                    knn_model.fit(collected_features, labels)
                    labeler = kmeans
            except Exception as e:
                knn_model = None  # Training failed, continue without classification
                st.warning(f"Model training failed: {str(e)}")

        # ===================================
        # Phase 3: Process Current Frame
        # ===================================
        # Apply full detection pipeline to current frame
        out_frame, mask = process_frame(frame, bg_subtractor, knn_model, labeler, trackers, settings)

        # Convert BGR (OpenCV) to RGB (Streamlit)
        orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)

        # Display frames side-by-side
        display_w = min(800, frame.shape[1])  # Limit width for performance
        orig_slot.image(orig_rgb, caption="Original", width=display_w)
        proc_slot.image(out_rgb, caption="Detection Output", width=display_w)

        # Control processing speed (FPS limiting)
        time.sleep(frame_time)

    # Cleanup
    cap.release()

    st.sidebar.markdown("Processing stopped.")


# ===================================
# Application Entry Point
# ===================================
if __name__ == '__main__':
    main()
