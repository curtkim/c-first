import cv2

cap = cv2.VideoCapture(0)   #ignore the errors
cap.set(3, 1920)             #Set the width important because the default will timeout ignore the error or false response
cap.set(4, 1080)             #Set the height ignore the errors
r, frame = cap.read()
cv2.imwrite("test.jpg", frame)
