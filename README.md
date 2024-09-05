---

# 🦋 Butterfly Object Detection with YOLOv8

This project aims to detect butterflies in images using **YOLOv8** object detection model. The dataset includes images of butterflies, annotated for object detection. The model is trained using a custom dataset and is capable of accurately identifying butterflies in images and videos. This project includes setup instructions, data preparation, training the model, and inference on both images and videos. Additionally, the project contains label normalization and dataset cleaning scripts.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Validation](#validation)
- [Inference on Images](#inference-on-images)
- [Inference on Videos](#inference-on-videos)
- [Label Normalization](#label-normalization)
- [Dataset Cleaning](#dataset-cleaning)
- [Results](#results)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

---


## Project Overview

This project utilizes **YOLOv8 (You Only Look Once version 8)** for object detection. The goal is to detect butterfly species in images with high accuracy. YOLOv8 is the latest version of the YOLO series and is optimized for faster training and better performance on object detection tasks.

---

## Dataset Structure

The dataset follows a common structure used in YOLOv8 for object detection tasks:

```bash
datasets/
├── coco8/
    ├── images/
    │   ├── train/          # Training images
    │   ├── val/            # Validation images
    │   └── test/           # Test images
    ├── labels/
    │   ├── train/          # Training labels (YOLO format)
    │   ├── val/            # Validation labels (YOLO format)
    │   └── test/           # Test labels (YOLO format)
```

The annotations are stored in **YOLO format** with the class "Butterfly" labeled as class 0.

---

## Installation

To get started with the project, follow the instructions below:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/butterfly-detection-yolov8.git
cd butterfly-detection-yolov8
```

### 2. Install Dependencies

You need Python 3.8+ and `pip` to install the necessary dependencies:

```bash
pip install ultralytics opencv-python pillow
```

---

## Training the Model

You can start training the YOLOv8 model on the butterfly dataset. Make sure that the dataset is structured properly and you have the required images and labels.

### 1. Training Command

Use the following command to train the YOLOv8 model:

```bash
from ultralytics import YOLO

# Initialize model
model = YOLO("yolov8n.pt")  # Using the nano model, can switch to larger models like yolov8s, yolov8m, etc.

# Train the model
model.train(data="/content/datasets/coco8/data.yaml", epochs=100, imgsz=640, batch=16)
```

- `data.yaml`: The configuration file specifying the dataset path.
- `epochs`: The number of epochs to train.
- `imgsz`: The image size (e.g., 640x640).
- `batch`: Batch size during training.

---

## Validation

After training, you can evaluate the model's performance on the validation dataset using the following command:

```bash
# Validate the trained model
model.val(data="/content/datasets/coco8/data.yaml")
```

This will evaluate the model on the validation images and output metrics like **Precision (P)**, **Recall (R)**, **mAP (mean Average Precision)**, etc.

---

## Inference on Images

Once the model is trained, you can test it on new images using the following command:

```python
# Inference on a single image
results = model('/path/to/your/image.jpg')

# Display results
results.show()
```

The image will be processed, and the detected butterflies will be annotated with bounding boxes and labels.

---

## Inference on Videos

To perform inference on a video file, you can use the following Python code:

```python
import cv2
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# Load the trained YOLOv8 model
model = YOLO('/content/runs/detect/train/weights/best.pt')

# Open the video file
cap = cv2.VideoCapture('/path/to/video.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection on the frame
    results = model(frame)
    
    # Annotate frame with detection
    annotated_frame = results[0].plot()

    # Display the frame
    cv2_imshow(annotated_frame)

cap.release()
cv2.destroyAllWindows()
```

This will process the video frame by frame and display the annotated results.

---

## Label Normalization

This script checks the label files to normalize the coordinates and ensures they are within valid ranges:

```python
import os
from PIL import Image

# Paths to your images and labels
image_dir = "/content/datasets/coco8/images/val"
label_dir = "/content/datasets/coco8/labels/val"

def normalize_labels(image_dir, label_dir):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        image_name_jpg = label_file.replace('.txt', '.jpg')
        image_name_png = label_file.replace('.txt', '.png')

        # Determine image file extension
        if os.path.exists(os.path.join(image_dir, image_name_jpg)):
            image_path = os.path.join(image_dir, image_name_jpg)
        elif os.path.exists(os.path.join(image_dir, image_name_png)):
            image_path = os.path.join(image_dir, image_name_png)
        else:
            print(f"Image file not found for label {label_file}")
            continue

        # Open image to get dimensions
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        new_lines = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = parts[0]
                    x1 = float(parts[1])
                    y1 = float(parts[2])
                    x2 = float(parts[3])
                    y2 = float(parts[4])

                    # Check if coordinates are pixel values
                    if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
                        # Calculate bounding box dimensions
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        x_center = x1 + (bbox_width / 2)
                        y_center = y1 + (bbox_height / 2)

                        # Normalize coordinates
                        x_center_norm = x_center / img_width
                        y_center_norm = y_center / img_height
                        bbox_width_norm = bbox_width / img_width
                        bbox_height_norm = bbox_height / img_height

                        # Ensure normalized coordinates are between 0 and 1
                        if not (0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1 and
                                0 <= bbox_width_norm <= 1 and 0 <= bbox_height_norm <= 1):
                            print(f"Normalized coordinates out of bounds in {label_path}: {line.strip()}")
                            continue

                        new_line = f"{class_id} {x_center_norm} {y_center_norm} {bbox_width_norm} {bbox_height_norm}\n"
                        new_lines.append(new_line)
                    else:
                        # Coordinates are already normalized
                        new_lines.append(line)
                else:
                    print(f"Incorrect label format in {label_path}: {line.strip()}")
                    continue

        # Write normalized labels back to file
        with open(label_path, 'w') as f:
            f.writelines(new_lines)

    print("Label normalization complete!")

normalize_labels(image_dir, label_dir)
```

---

## Dataset Cleaning

This script inspects and cleans up the dataset, including removing corrupt images and fixing label files.

```python
import os
import cv2

# Define the path to your dataset (update this with the actual path)
image_dir = "/content/datasets/coco8/images/train"
label_dir = "/content/datasets/coco8/labels/train"

# Mapping class names (e.g., 'Butterfly') to numeric IDs
class_mapping = {
    'Butterfly': 0,
    # Add other mappings here if needed
}

def inspect_label_file(label_path):
    """
    Inspect label files and log issues. Return True if file can be fixed, False otherwise.
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4 or len(parts) > 5:
                print(f"Incorrect label format: {label

_path}")
                return False  # Invalid format
            class_id = int(parts[0])
            if class_id not in class_mapping.values():
                print(f"Invalid class ID: {label_path}")
                return False  # Invalid class ID
        return True

def clean_dataset(image_dir, label_dir):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        image_name_jpg = label_file.replace('.txt', '.jpg')
        image_name_png = label_file.replace('.txt', '.png')

        # Remove empty label files
        if os.stat(label_path).st_size == 0:
            print(f"Empty label file removed: {label_path}")
            os.remove(label_path)
            continue

        # Verify if corresponding image exists
        image_path = None
        if os.path.exists(os.path.join(image_dir, image_name_jpg)):
            image_path = os.path.join(image_dir, image_name_jpg)
        elif os.path.exists(os.path.join(image_dir, image_name_png)):
            image_path = os.path.join(image_dir, image_name_png)

        if image_path is None:
            print(f"No corresponding image for label: {label_path}")
            continue

        # Remove corrupt images
        try:
            img = cv2.imread(image_path)
            if img is None or img.size == 0:
                print(f"Corrupt image removed: {image_path}")
                os.remove(image_path)
                os.remove(label_path)
                continue
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            continue

        # Inspect label files
        if not inspect_label_file(label_path):
            print(f"Label file issues found: {label_path}")
            os.remove(label_path)
            os.remove(image_path)

clean_dataset(image_dir, label_dir)
```

---

## Results

### Model Performance Metrics

| Metric   | Score  |
| -------- | ------ |
| Precision| 81.4%  |
| Recall   | 68.2%  |
| mAP@50   | 78.0%  |
| mAP@50-95| 40.5%  |

These scores indicate good detection accuracy, though further tuning may be needed for specific scenarios.

The following graph shows the training and validation loss and performance metrics over 100 epochs for the object detection task:

![results](https://github.com/user-attachments/assets/8684314a-4636-4b79-8330-f193e6e0dda1)

```markdown
## Training and Validation Results

- **train/box_loss**: Loss associated with bounding box predictions during training.
- **train/cls_loss**: Loss associated with classification predictions during training.
- **train/dfl_loss**: Distribution Focal Loss (DFL) during training.
- **val/box_loss**: Loss associated with bounding box predictions during validation.
- **val/cls_loss**: Loss associated with classification predictions during validation.
- **val/dfl_loss**: Distribution Focal Loss (DFL) during validation.
- **metrics/precision(B)**: Precision metrics for bounding box predictions.
- **metrics/recall(B)**: Recall metrics for bounding box predictions.
- **metrics/mAP50(B)**: Mean Average Precision (mAP) at IoU 50%.
- **metrics/mAP50-95(B)**: Mean Average Precision (mAP) across IoU thresholds 50% to 95%.
```

---

### Explanation of the Results(If you are not interested you are free to skip 😉)

The graph above illustrates the training and validation process over 100 epochs for the object detection task using YOLOv8. Each plot showcases key loss and metric trends as the model improves during training and validation:

1. **train/box_loss**: This curve represents the loss related to bounding box predictions during training. A decreasing trend indicates that the model is getting better at accurately predicting object locations.
   
2. **train/cls_loss**: This is the classification loss during training, which reflects how well the model is classifying objects into the correct categories. The steady decline shows that the model is learning to make more accurate predictions over time.
   
3. **train/dfl_loss**: The Distribution Focal Loss (DFL) during training measures how well the model is predicting the bounding box distributions. A lower value suggests improved localization precision.

4. **val/box_loss**: This plot shows the bounding box loss during validation. The trend is similar to the training box loss, suggesting that the model generalizes well to unseen data.
   
5. **val/cls_loss**: Classification loss on the validation set. A decreasing trend is a sign that the model is not overfitting and can classify objects in validation images correctly.

6. **val/dfl_loss**: Similar to the training DFL loss but calculated on the validation set. A smooth decrease shows better object localization for unseen images.
   
7. **metrics/precision(B)**: Precision measures how many of the predicted bounding boxes were actually correct. A value closer to 1 indicates that the model is very precise and makes few false-positive predictions.
   
8. **metrics/recall(B)**: Recall measures how many of the actual objects the model detected correctly. An increasing trend shows that the model is identifying more objects correctly as training progresses.

9. **metrics/mAP50(B)**: The Mean Average Precision (mAP) at an Intersection over Union (IoU) threshold of 50%. It is a commonly used metric for object detection, and a steady increase shows better performance.

10. **metrics/mAP50-95(B)**: This is the mean average precision calculated over different IoU thresholds (50% to 95%). This metric reflects how well the model is performing across various levels of detection strictness.

### Key Takeaways:
- The **train/box_loss**, **train/cls_loss**, and **train/dfl_loss** curves show steady improvement, which suggests that the model is learning well.
- The validation losses also decrease, indicating that the model is generalizing well on unseen data.
- **Precision** and **Recall** improve over time, reflecting better detection accuracy and coverage.
- The increase in **mAP50** and **mAP50-95** shows that the model is getting better at detecting objects across a range of IoU thresholds.

---

Including this explanation will not only make your results more understandable to others, but also demonstrate your deep understanding of the model training process and evaluation metrics.

### Example Detections:

![Sample Butterfly Detection](path/to/sample_image.jpg)

---

## Contributing

Feel free to submit issues or pull requests if you would like to contribute to this project. Any improvements, bug fixes, or performance optimizations are welcome!

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
