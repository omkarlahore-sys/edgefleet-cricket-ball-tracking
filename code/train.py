from ultralytics import YOLO

def train():
    """
    Optional training script for reproducibility.
    Not required to run inference.
    """

    model = YOLO("yolov8s.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=960,
        batch=8
    )

if __name__ == "__main__":
    train()
