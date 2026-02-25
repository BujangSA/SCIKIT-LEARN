import pickle
import cv2
import mediapipe as mp
import numpy as np


with open("model.pickle", "rb") as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


            fingers_open = 0

            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                fingers_open += 1
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
                fingers_open += 1
            if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
                fingers_open += 1
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
                fingers_open += 1

            if fingers_open >= 3:
                label = "BUKA"
                color = (0, 255, 0)
            else:
                label = "TUTUP"
                color = (0, 0, 255)


        cv2.putText(
            frame,
            label,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3
        )

    cv2.imshow("Deteksi Telapak Tangan", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()