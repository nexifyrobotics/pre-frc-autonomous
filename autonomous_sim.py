import cv2
import numpy as np
from pupil_apriltags import Detector

cap = cv2.VideoCapture(0)

detector = Detector(families="tag36h11")

FOCAL_LENGTH = 700
TAG_SIZE = 0.16

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera bulunamadı.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)

    for r in results:
        (ptA, ptB, ptC, ptD) = np.int32(r.corners)
        cv2.polylines(frame, [np.array([ptA, ptB, ptC, ptD])], True, (0, 255, 0), 2)
        center_x, center_y = map(int, r.center)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        perceived_width = np.linalg.norm(ptA - ptB)
        if perceived_width > 0:
            distance = (TAG_SIZE * FOCAL_LENGTH) / perceived_width
        else:
            distance = None

        frame_center_x = frame.shape[1] / 2
        offset_x = center_x - frame_center_x

        if distance:
            if abs(offset_x) < 50:
                if distance > 60:
                    action = "İLERİ GİT"
                else:
                    action = "DUR"
            elif offset_x > 50:
                action = "SAGA DON "
            else:
                action = "SOLA DON"
        else:
            action = "Tag bulunamadı."

        cv2.putText(frame, f"{action}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        print(f"Tag {r.tag_id} | Mesafe: {distance:.1f} cm | Offset: {offset_x:.1f} | Aksiyon: {action}")

    cv2.imshow("Otonom Simulasyon", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()