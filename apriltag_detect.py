import cv2
import numpy as np
from pupil_apriltags import Detector

detector = Detector(families="tag36h11")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    for tag in tags:
        (ptA, ptB, ptC, ptD) = tag.corners
        ptA, ptB, ptC, ptD = np.int32([ptA, ptB, ptC, ptD])
        cv2.polylines(frame, [np.array([ptA, ptB, ptC, ptD])], True, (0,255,0), 2)
        cv2.putText(frame, f"ID: {tag.tag_id}", (ptA[0], ptA[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        print(f"Tag {tag.tag_id} - Center: {tag.center} - Rotation: {tag.pose_R}")

    cv2.imshow("AprilTag", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()