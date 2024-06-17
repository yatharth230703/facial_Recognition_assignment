import cv2
import numpy as np
import sys
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO(r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\yolov8n-oiv7.pt')  
# Initialize Deep SORT
max_cosine_distance = 0.4
nn_budget = None
model_filename = r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\mars-small128.pb' 
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Function to apply Non-Maximum Suppression (NMS)
def non_max_suppression(boxes, scores, threshold=0.3):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
    return indices

# Initialize video capture
cap = cv2.VideoCapture(r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\theboys_test_reduced_faces.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    height, width, _ = frame.shape

    # YOLOv8 detection
    results = yolo_model(frame)
    detections_data = results[0].boxes.data.cpu().numpy()  

    yolo_boxes = []
    yolo_scores = []
    face_class_id = 1  

    print("Detections:")
    for detection in detections_data:
        class_id = int(detection[5])
        confidence = detection[4]
        print(f"Class ID: {class_id}, Confidence: {confidence}, Bounding Box: {detection[:4]}")  
        if confidence > 0.5 and class_id == face_class_id:
            x1, y1, x2, y2 = detection[:4]
            box_width = x2 - x1
            box_height = y2 - y1
            yolo_boxes.append([int(x1), int(y1), int(box_width), int(box_height)])
            yolo_scores.append(float(confidence))

    print(f"Filtered YOLO Boxes: {yolo_boxes}")
    print(f"Filtered YOLO Scores: {yolo_scores}")

    # Apply Non-Maximum Suppression
    indices = non_max_suppression(yolo_boxes, yolo_scores)
    print(f"Indices after NMS: {indices}")

    detections = []

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = yolo_boxes[i]
            bbox = [x, y, w, h]
            detections.append(Detection(bbox, yolo_scores[i], encoder(frame, [bbox])[0]))

    print(f"Detections after NMS and Encoding: {detections}")

    # Update tracker with current detections
    tracker.predict()
    tracker.update(detections)

    # Draw tracked bounding boxes
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        track_id = track.track_id
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Face Detection and Deep SORT Tracking', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
