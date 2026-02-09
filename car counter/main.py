from ultralytics import YOLO
import cv2
from sort import Sort
import numpy as np


model = YOLO("yolov8n.pt")

capture = cv2.VideoCapture('cars2.mp4')  
mask = cv2.imread('mask.png')  
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  
countLiner = [100, 297, 1200, 297]  # Define the counting line (x1, y1, x2, y2)
counter = []
while True:
    success, img = capture.read()
    imgRegion = cv2.bitwise_and(img, mask)  # Apply the mask to the frame
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))  # Initialize an empty array for detections

    for r in results:

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ['car', 'truck', 'bus', 'motorcycle']:  # Filter for vehicle classes
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                # cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                current = np.array([[x1, y1, x2, y2, conf]])
                detections = np.vstack((detections, current))  # Append current detection to the array
    resultsTracker = tracker.update(detections)
    cv2.line(img, (countLiner[0], countLiner[1]), (countLiner[2], countLiner[3]), (255, 0, 255), 2)
    for x1, y1, x2, y2, _id in resultsTracker:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"ID: {_id}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        if countLiner[0] < cx < countLiner[2] and countLiner[1] - 15 < cy < countLiner[3] + 15:
            if _id not in counter:
                counter.append(_id)
                cv2.line(img, (countLiner[0], countLiner[1]), (countLiner[2], countLiner[3]), (0, 255, 0), 5)
    cv2.putText(img, f"Count: {len(counter)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)            
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)

