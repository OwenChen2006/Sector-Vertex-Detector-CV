# SCRIPT 2: YOLOv8 INFERENCE
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
# --- Configuration ---
# Path to your BEST trained weights file
TRAINED_MODEL_PATH = r"C:\Users\CalidarTeam\OneDrive - Calidar Medical\Intern\Owen\BlockData_Processed\dataset\runs\detect\train3\weights\best.pt"
# Path to the new, unmarked image you want to test
IMAGE_TO_TEST_PATH = r"C:\Users\CalidarTeam\OneDrive - Calidar Medical\Intern\Owen\BlockData_Processed\dataset\train\images\Case_025_png.rf.ff9cc30ffdc2cc5d25506da23068acbb.jpg"
CONFIDENCE_THRESHOLD = 0.49

def main():
    """Runs inference using the fine-tuned YOLOv8 model to find the center."""
    print("--- Loading Fine-Tuned YOLOv8 Model ---")
    model = YOLO(TRAINED_MODEL_PATH)

    print(f"\n--- Running Inference on {os.path.basename(IMAGE_TO_TEST_PATH)} ---")
    # Run prediction
    results = model(IMAGE_TO_TEST_PATH)

    # Load the image for visualization
    img = cv2.imread(IMAGE_TO_TEST_PATH)
    
    # Process the results
    for result in results:
        boxes = result.boxes.cpu().numpy() # Get boxes on CPU in numpy format
        if len(boxes) > 0:
            print(f"Found {len(boxes)} potential objects.")
            # Assume the highest confidence box is the one we want
            best_box_index = np.argmax(boxes.conf)
            box = boxes[best_box_index]

            if box.conf > CONFIDENCE_THRESHOLD:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Calculate center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                print("\n" + "="*50)
                print("                INFERENCE REPORT")
                print("="*50)
                print(f"Detected 'center_pattern' with confidence: {box.conf[0]:.4f}")
                print(f"Predicted Center Coordinate: ({center_x}, {center_y})")
                print("="*50)

                # Draw the bounding box and center point for visualization
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
            else:
                print("No object found with confidence above the threshold.")
        else:
            print("No objects detected in the image.")

    # Display the final image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.title("YOLOv8 Center Detection Result")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()