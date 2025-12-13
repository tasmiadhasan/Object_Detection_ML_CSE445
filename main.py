import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# =====================
# CONFIG
# =====================
VIDEO_PATH = "video.mp4"   #any demo video
MIN_AREA = 800             # ignore small contours
MAX_OBJECTS = 6            # expected max moving objects
HISTORY_FRAMES = 2         # for velocity estimation

# =====================
# INITIALIZE
# =====================
cap = cv2.VideoCapture(VIDEO_PATH)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

prev_centroids = []
records = []
frame_id = 0

# =====================
# MAIN LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    contours, _ = cv2.findContours(
        fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    centroids = []

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        detections.append((x, y, w, h))
        centroids.append([cx, cy])

    centroids = np.array(centroids)

    # =====================
    # FEATURE VECTOR
    # =====================
    features = []

    if len(centroids) > 0:
        for i, (cx, cy) in enumerate(centroids):
            if len(prev_centroids) >= HISTORY_FRAMES:
                vx = cx - prev_centroids[-1][i][0]
                vy = cy - prev_centroids[-1][i][1]
            else:
                vx, vy = 0, 0

            features.append([cx, cy, vx, vy])

        features = np.array(features)

        # =====================
        # k-MEANS CLUSTERING
        # =====================
        K = min(MAX_OBJECTS, len(features))
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)

        # =====================
        # OUTPUT + VISUALIZATION
        # =====================
        for i, label in enumerate(labels):
            x, y, w, h = detections[i]
            cx, cy = centroids[i]

            records.append({
                "frame": frame_id,
                "object_id": int(label),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "cx": int(cx),
                "cy": int(cy)
            })

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(
                frame, f"ID {label}",
                (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,255,0), 2
            )

    prev_centroids.append(centroids.tolist())
    if len(prev_centroids) > HISTORY_FRAMES:
        prev_centroids.pop(0)

    cv2.imshow("k-Means Multi-Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# =====================
# SAVE OUTPUT
# =====================
df = pd.DataFrame(records)
df.to_csv("kmeans_tracking_output.csv", index=False)

print("Tracking data saved to kmeans_tracking_output.csv")
