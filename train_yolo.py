from ultralytics import YOLO

# Load your model (pretrained or previous checkpoint)
model = YOLO(r"C:\Users\Raiz\emtech\flame_detector\runs\detect\train4\weights\best.pt")

# Start training
model.train(
    data=r"fire.v1i.yolov8\data.yaml",  # relative path from train_yolo.py
    epochs=1,              # number of training epochs
    imgsz=640,              # image size
    resume=False           # resume training if a previous run exists
)
