from ultralytics import YOLO
import os

# 1. Pastikan script ni tahu dia kat mane (dalam SSD)
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

def main():
    # 2. Load model 'Nano' (paling ringan, sesuai untuk laptop)
    # Dia akan download file 'yolov8n.pt' secara automatik masa first run
    model = YOLO('yolov8n.pt')

    # 3. Start training
    # Ganti rock-paper-scissors dengan nama folder dataset korang kalau lain
    model.train(
        data='Rock Paper Scissors SXSW.v14i.yolov8/data.yaml',  # Path ke data.yaml
        epochs=5,
        imgsz=640,
        batch=4,
        device='cpu',
        workers=0
    )

if __name__ == '__main__':
    main()