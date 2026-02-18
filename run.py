from ultralytics import YOLO
import cv2

# 1. Load model yang dah siap 'train' tadi
# Ganti path ni ikut folder 'runs' kau
model = YOLO('runs/detect/train/weights/best.pt') 

# 2. Buka webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 3. Suruh model teka apa dia nampak
        results = model(frame, conf=0.5) # conf=0.5 maksudnya dia kena 50% yakin baru dia tunjuk

        # 4. Lukis kotak result kat skrin
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Real-Time", annotated_frame)

        # Tekan 'q' untuk berhenti
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()