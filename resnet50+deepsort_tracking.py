import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import cv2
import numpy as np
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

# Load the pre-trained Faster R-CNN ResNet-50 model with 2 classes
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
state_dict = torch.load(r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\face_tracking_resnet50.pth')
model.load_state_dict(state_dict)
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

def detect_faces(model, frame):
    # Convert frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Apply transformations
    input_image = transform(image).unsqueeze(0)

    # Perform detection
    with torch.no_grad():
        outputs = model(input_image)

    # Extract bounding boxes and scores
    bounding_boxes = []
    scores = outputs[0]['scores'].cpu().numpy()
    boxes = outputs[0]['boxes'].cpu().numpy()
    for box, score in zip(boxes, scores):
        if score > 0.5:  # Confidence threshold
            bounding_boxes.append(box)

    return bounding_boxes

# Initialize Deep SORT
max_cosine_distance = 0.4
nn_budget = None
model_filename = r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\mars-small128.pb'  
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    height, width, _ = frame.shape

    # Detect faces using the Faster R-CNN model
    bounding_boxes = detect_faces(model, frame)
    
    # Extract features for each bounding box
    features = encoder(frame, bounding_boxes)

    # Create detections
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bounding_boxes, features)]

    # Update tracker
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
    cv2.imshow('Face Detection and Tracking', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


