import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

start_time = 0
current_time = 0

while True:
    success, img = capture.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1 / (current_time - start_time)
    start_time = current_time
    cv.putText(img, 'fps: '+str(int(fps)), (5, 30), cv.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)


    cv.imshow("Video Capture", img)
    cv.waitKey(1)