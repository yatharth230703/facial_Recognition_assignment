import cv2
import numpy as np
import sys

# Add the path to the SORT directory
sys.path.append(r'C:\Users\Yatharth\sort')  
from sort import Sort

# Load YOLO model
yolo_net = cv2.dnn.readNet(r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\yolov3-wider_16000.weights', r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\yolov3-face.cfg')
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Initialize SORT tracker
tracker = Sort()

# Function to apply Non-Maximum Suppression (NMS)
def non_max_suppression(boxes, scores, threshold=0.3):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
    return indices

# Initialize video capture
cap = cv2.VideoCapture(r"C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\theboys_test_4x.mp4")

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

    yolo_faces = []
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
                yolo_faces.append([x, y, int(box_width), int(box_height)])
                yolo_scores.append(float(confidence))

    if yolo_faces:
        indices = non_max_suppression(yolo_faces, yolo_scores)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y, w, h) = yolo_faces[i]
                detections.append([x, y, x + w, y + h, yolo_scores[i]])

        # Update tracker with current detections
        trackers = tracker.update(np.array(detections))

        # Draw tracked bounding boxes
        for d in trackers:
            x1, y1, x2, y2, track_id = d
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Face Detection and Tracking', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
