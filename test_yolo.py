from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\Raiz\emtech\flame_detector\runs\detect\train12\weights\best.pt")

# Run prediction (webcam as source 0)
model.predict(
    source=0,       # 0 = default webcam
    show=True,      # display the live prediction window
    save=True       # save output video in runs/detect/predict
)
