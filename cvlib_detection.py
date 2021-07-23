import cvlib
from cvlib.object_detection import draw_bbox
import cv2

cap = cv2.VideoCapture(0)  # capture from webcam

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print('Capturing is stopped')
        break

    faces, confidence = cvlib.detect_face(image, threshold=0.5, enable_gpu=False)
    labels = [str(label) for label in range(len(faces))]
    output = draw_bbox(image, faces, labels=['person'], confidence=confidence, colors=None, write_conf=True)

    cv2.imshow('output', output)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
