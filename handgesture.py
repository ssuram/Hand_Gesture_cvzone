import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
mpdraw = mp.solutions.drawing_utils
hands = mphands.Hands()
prevtime = 0
curtime = 0
while True:
    success, img = cap.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgrgb)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handsmp in results.multi_hand_landmarks:
            for id, lm in enumerate(handsmp.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
            mpdraw.draw_landmarks(img, handsmp, mphands.HAND_CONNECTIONS)

    curtime = time.time()
    fps = 1 / (curtime - prevtime)
    prevtime = curtime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255, 3))
    cv2.imshow("image", img)
    cv2.waitKey(1)