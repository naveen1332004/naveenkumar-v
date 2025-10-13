from ultralytics import YOLO
import cv2
import numpy as np

# ---- Load YOLOv8 model ----
model = YOLO("yolov8n.pt")

# ---- Load image ----
image_path = r"C:\Users\navee\OneDrive\Documents\APPLY 01\MSC DS\001 CV LAB\bus.jpg"
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Image not found at given path!")

# ---- Run detection ----
results = model(img)

# ---- Extract detections ----
detected_boxes = []
detected_classes = []
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy()
    for box, cls in zip(boxes, clss):
        detected_boxes.append(list(map(int, box)))
        detected_classes.append(int(cls))

        # Draw detections on image
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ---- Ground Truth (replace with your real annotations) ----
# Format: [x_min, y_min, x_max, y_max, class_id]
ground_truths = [
    [50, 50, 200, 400, 0],   # Example: Person
    [300, 100, 450, 250, 2]  # Example: Car
]

# ---- IOU function ----
def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# ---- Evaluation ----
TP, FP, FN = 0, 0, 0
matched_detections = set()

for gt in ground_truths:
    gt_box, gt_cls = gt[:4], gt[4]
    matched = False
    for i, det in enumerate(detected_boxes):
        if i in matched_detections:
            continue
        det_cls = detected_classes[i]
        if det_cls == gt_cls and iou(gt_box, det) > 0.5:  # class + IOU check
            TP += 1
            matched = True
            matched_detections.add(i)
            break
    if not matched:
        FN += 1

FP = len(detected_boxes) - len(matched_detections)

accuracy = TP / (TP + FP) if (TP + FP) > 0 else 0
miss_rate = FN / (TP + FN) if (TP + FN) > 0 else 0

print(f"Accuracy: {accuracy:.2f}")
print(f"Miss Rate: {miss_rate:.2f}")

# ---- Show output ----
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()