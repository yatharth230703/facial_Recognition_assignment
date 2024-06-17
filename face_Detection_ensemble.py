import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection (BlazeFace)
mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize another face detection model (Haar Cascade for this example)
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to apply Non-Maximum Suppression (NMS)
def non_max_suppression(boxes, scores, threshold=0.3):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
    return indices

# Initialize webcam
cap = cv2.VideoCapture(r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\final_test_file.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

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

    # Combine results
    all_faces = blaze_faces + haar_faces
    all_scores = blaze_scores + haar_scores

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
