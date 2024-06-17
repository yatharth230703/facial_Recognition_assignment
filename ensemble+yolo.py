import cv2
import mediapipe as mp
import numpy as np
# Initialize MediaPipe Face Detection (BlazeFace)
mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load YOLO model
yolo_net = cv2.dnn.readNet('face.weights', 'face.cfg')
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Function to apply Non-Maximum Suppression (NMS)
def non_max_suppression(boxes, scores, threshold=0.3):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
    return indices

# Initialize video capture
cap = cv2.VideoCapture(r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\theboys_test_4x.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    height, width, _ = frame.shape

    # Convert the BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # BlazeFace detection
    results = mp_face_detection.process(rgb_frame)
    blaze_faces = []
    blaze_scores = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            xmin = int(bboxC.xmin * width)
            ymin = int(bboxC.ymin * height)
            box_width = int(bboxC.width * width)
            box_height = int(bboxC.height * height)
            blaze_faces.append([xmin, ymin, box_width, box_height])
            blaze_scores.append(detection.score[0])

    # Haar Cascade detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar_faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    haar_faces = [list(face) for face in haar_faces]
    haar_scores = [1.0] * len(haar_faces) 

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

    # Combine results
    all_faces = blaze_faces + haar_faces + yolo_faces
    all_scores = blaze_scores + haar_scores + yolo_scores

    # NMS
    if all_faces:
        indices = non_max_suppression(all_faces, all_scores)

        # Draw final bounding boxes
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y, w, h) = all_faces[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Ensembled Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
