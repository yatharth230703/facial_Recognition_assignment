import cv2
import numpy as np
import tensorflow as tf
import sys
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

# Load YOLO model
##Trylater : yolov3-wider_16000
yolo_net = cv2.dnn.readNet('yolov3-wider_16000.weights', 'yolov3-face.cfg')
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

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
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    height, width, _ = frame.shape

    # YOLO detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    yolo_outputs = yolo_net.forward(output_layers)

    yolo_boxes = []
    yolo_scores = []

    for output in yolo_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, box_width, box_height) = box.astype("int")
                x = int(centerX - (box_width / 2))
                y = int(centerY - (box_height / 2))
                yolo_boxes.append([x, y, int(box_width), int(box_height)])
                yolo_scores.append(float(confidence))

    # NMS
    indices = non_max_suppression(yolo_boxes, yolo_scores)
    detections = []

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = yolo_boxes[i]
            bbox = [x, y, w, h]
            detections.append(Detection(bbox, yolo_scores[i], encoder(frame, [bbox])[0]))

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
    cv2.imshow('YOLO Face Detection and Deep SORT Tracking', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
