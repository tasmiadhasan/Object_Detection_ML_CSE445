"""
Object Detection Model
KNN + Kalman Filter Implementation
"""

import cv2
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from filterpy.kalman import KalmanFilter


class ObjectDetectionModel:
    """
    Object Detection using KNN classification and Kalman Filter tracking
    """
    
    def __init__(self, 
                 n_neighbors=3, 
                 n_clusters=3, 
                 min_area=400,
                 bg_var_threshold=25,
                 use_kalman=True,
                 collect_frames=30):
        """
        Initialize the object detection model
        
        Parameters:
        - n_neighbors: Number of neighbors for KNN classifier
        - n_clusters: Number of clusters for KMeans
        - min_area: Minimum contour area to consider
        - bg_var_threshold: Background subtractor variance threshold
        - use_kalman: Enable/disable Kalman filter tracking
        - collect_frames: Number of frames to collect before training
        """
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.min_area = min_area
        self.bg_var_threshold = bg_var_threshold
        self.use_kalman = use_kalman
        self.collect_frames = collect_frames
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=bg_var_threshold, 
            detectShadows=False
        )
        
        # Model components
        self.knn_model = None
        self.labeler = None
        self.trackers = {}
        
        # Feature collection
        self.collected_features = []
        self.frame_count = 0
        self.is_trained = False
    
    def create_kalman(self):
        """
        Create a Kalman Filter for object tracking
        4 states: x, y, vx, vy
        2 measurements: x, y
        """
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
    
    def extract_features(self, contour, frame_shape):
        """
        Extract features from a contour
        Features: normalized centroid x,y, normalized area, aspect ratio
        """
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        cx = x + w / 2.0
        cy = y + h / 2.0
        fw, fh = frame_shape[1], frame_shape[0]
        
        return [
            cx / fw,  # normalized x
            cy / fh,  # normalized y
            area / (fw * fh),  # normalized area
            float(w) / float(h + 1e-6)  # aspect ratio
        ]
    
    def train_model(self):
        """
        Train KNN model using collected features
        Uses KMeans for clustering to create pseudo-labels
        """
        if len(self.collected_features) == 0:
            print("No features collected for training")
            return False
        
        try:
            # Use KMeans to create pseudo-labels
            n_clusters = min(self.n_clusters, max(1, len(self.collected_features)))
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            labels = kmeans.fit_predict(self.collected_features)
            
            # Train KNN classifier
            self.knn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            self.knn_model.fit(self.collected_features, labels)
            self.labeler = kmeans
            self.is_trained = True
            
            print(f"Model trained with {len(self.collected_features)} features and {n_clusters} clusters")
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def detect_objects(self, frame):
        """
        Detect objects in a frame
        Returns list of detections with bounding boxes, features, and labels
        """
        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Background subtraction
        fgmask = self.bg_subtractor.apply(blur)
        _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            feat = self.extract_features(cnt, frame.shape)
            centroid = (int(x + w/2), int(y + h/2))
            
            detection = {
                'bbox': (x, y, w, h),
                'feature': feat,
                'centroid': centroid,
                'area': area
            }
            
            # Classify if model is trained
            if self.is_trained and self.knn_model is not None:
                pred = self.knn_model.predict([feat])[0]
                detection['label'] = pred
            else:
                detection['label'] = -1
            
            detections.append(detection)
        
        return detections, thresh
    
    def update_trackers(self, detections):
        """
        Update Kalman filter trackers for detected objects
        """
        if not self.use_kalman:
            for det in detections:
                det['kf_centroid'] = det['centroid']
            return
        
        # Update trackers
        for det in detections:
            lbl = det['label']
            cx, cy = det['centroid']
            
            if lbl not in self.trackers:
                kf = self.create_kalman()
                kf.x = np.array([cx, cy, 0, 0], dtype=float)
                self.trackers[lbl] = {'kf': kf, 'last_seen': time.time()}
            else:
                self.trackers[lbl]['kf'].predict()
                self.trackers[lbl]['kf'].update(np.array([cx, cy]))
                self.trackers[lbl]['last_seen'] = time.time()
            
            # Use Kalman estimate for smoother tracking
            kf = self.trackers[lbl]['kf']
            det['kf_centroid'] = (int(kf.x[0]), int(kf.x[1]))
        
        # Remove stale trackers (not seen for 1.5 seconds)
        stale = [lbl for lbl, info in self.trackers.items() 
                 if time.time() - info['last_seen'] > 1.5]
        for lbl in stale:
            del self.trackers[lbl]
    
    def process_frame(self, frame):
        """
        Process a single frame and return annotated output
        
        Returns:
        - annotated_frame: Frame with bounding boxes and labels
        - mask: Binary mask showing detected motion
        - detections: List of detection dictionaries
        """
        self.frame_count += 1
        
        # Collect features for training if not trained yet
        if not self.is_trained and self.frame_count <= self.collect_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            fgmask = self.bg_subtractor.apply(blur)
            _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < self.min_area:
                    continue
                feat = self.extract_features(cnt, frame.shape)
                self.collected_features.append(feat)
            
            # Train model after collecting enough frames
            if self.frame_count == self.collect_frames:
                self.train_model()
        
        # Detect objects
        detections, mask = self.detect_objects(frame)
        
        # Update trackers
        self.update_trackers(detections)
        
        # Annotate frame
        annotated_frame = self.draw_detections(frame.copy(), detections)
        
        return annotated_frame, mask, detections
    
    def draw_detections(self, frame, detections, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes and labels on frame
        
        Parameters:
        - frame: Input frame to annotate
        - detections: List of detection dictionaries
        - color: BGR color tuple
        - thickness: Line thickness
        """
        for det in detections:
            x, y, w, h = det['bbox']
            cx, cy = det.get('kf_centroid', det['centroid'])
            label = det.get('label', -1)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Draw centroid
            cv2.circle(frame, (cx, cy), 3, color, -1)
            
            # Draw label
            cv2.putText(frame, f"ID:{label}", (x, y-6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def process_video(self, video_path, output_path=None, max_frames=None, 
                     display=False, bbox_color=(0, 255, 0), bbox_thickness=2):
        """
        Process an entire video file
        
        Parameters:
        - video_path: Path to input video
        - output_path: Path to save output video (optional)
        - max_frames: Maximum number of frames to process (optional)
        - display: Show output in window (requires GUI environment)
        - bbox_color: Color for bounding boxes (BGR)
        - bbox_thickness: Thickness of bounding boxes
        
        Returns:
        - stats: Dictionary with processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing statistics
        stats = {
            'total_frames_processed': 0,
            'total_detections': 0,
            'processing_time': 0
        }
        
        start_time = time.time()
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break
            
            # Process frame
            annotated_frame, mask, detections = self.process_frame(frame)
            
            # Draw with custom color
            annotated_frame = self.draw_detections(
                frame.copy(), detections, 
                color=bbox_color, thickness=bbox_thickness
            )
            
            # Update statistics
            stats['total_frames_processed'] += 1
            stats['total_detections'] += len(detections)
            
            # Write to output
            if writer:
                writer.write(annotated_frame)
            
            # Display if requested
            if display:
                cv2.imshow('Object Detection', annotated_frame)
                cv2.imshow('Mask', mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames, "
                      f"Detections: {len(detections)}")
        
        stats['processing_time'] = time.time() - start_time
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Frames processed: {stats['total_frames_processed']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        print(f"Average FPS: {stats['total_frames_processed']/stats['processing_time']:.2f}")
        
        return stats


# Example usage for presentation
if __name__ == '__main__':
    # Create model instance
    model = ObjectDetectionModel(
        n_neighbors=3,
        n_clusters=3,
        min_area=400,
        bg_var_threshold=25,
        use_kalman=True,
        collect_frames=30
    )
    
    # Example: Process a video
    # Replace 'input_video.mp4' with your video path
    video_path = 'input_video.mp4'
    output_path = 'output_video.mp4'
    
    # Process video (comment out if you don't have a video file)
    # stats = model.process_video(
    #     video_path=video_path,
    #     output_path=output_path,
    #     max_frames=None,  # Process all frames
    #     display=False,  # Set to True to show live display
    #     bbox_color=(0, 255, 0),  # Green
    #     bbox_thickness=2
    # )
    
    print("Model ready for presentation!")
    print("\nKey components:")
    print("- KNN Classifier for object classification")
    print("- Kalman Filter for smooth tracking")
    print("- Background subtraction for motion detection")
    print("- Feature extraction: centroid, area, aspect ratio")
