# SCRIPT 1: RE-TRAINING YOLOv8 WITHOUT MOSAIC
from ultralytics import YOLO

# --- Configuration ---
# Path to your data.yaml file
DATA_YAML_PATH = r"C:\Users\CalidarTeam\OneDrive - Calidar Medical\Intern\Owen\BlockData_Processed\dataset\data.yaml"
# We'll start from the same pre-trained model
PRETRAINED_MODEL = 'yolov8n.pt'
# Training parameters
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 4

def main():
    """Trains a YOLOv8 object detection model with mosaic augmentation DISABLED."""
    print("--- Loading Pre-trained YOLOv8 Model ---")
    model = YOLO(PRETRAINED_MODEL)

    print("\n--- Starting Fine-Tuning (MOSAIC DISABLED) on Custom XRD Dataset ---")
    # Train the model on your custom dataset
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        # --- THIS IS THE CRITICAL FIX ---
        # mosaic=0 disables the 4-image stitching augmentation.
        # The model will now train on your single images.
        mosaic=0,
        # You can add other standard augmentations if you wish
        degrees=10,    # Random rotations
        fliplr=0.5     # Random horizontal flips
    )

    print("\n--- Training Complete ---")
    # A new training run folder will be created (e.g., 'train8')
    # with the newly trained 'best.pt'
    metrics = model.val()
    print(f"Final Validation Metrics: {metrics.box.map50}")

if __name__ == '__main__':
    main()