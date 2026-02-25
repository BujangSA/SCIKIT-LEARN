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

labels_dict = {
0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',
8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',
20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'
}

# ===== VARIABEL SEMI STABIL =====
frame_counter = 0
update_interval = 7   # ganti jadi 5-10 sesuai kebutuhan
huruf_tampil = ""

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

            # NORMALISASI DATA
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            # DETEKSI BUKA/TUTUP
            fingers_open = 0
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                fingers_open += 1
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
                fingers_open += 1
            if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
                fingers_open += 1
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
                fingers_open += 1

            palm_label = "BUKA" if fingers_open >= 3 else "TUTUP"

        # PREDIKSI MODEL
        prediction = model.predict([np.array(data_aux)])
        huruf_prediksi = labels_dict[int(prediction[0])]

        # ===== UPDATE SETIAP BEBERAPA FRAME =====
        frame_counter += 1
        if frame_counter >= update_interval:
            huruf_tampil = huruf_prediksi
            frame_counter = 0

        # TAMPILKAN HURUF BESAR
        cv2.putText(frame, huruf_tampil,
                    (250, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,
                    (0, 255, 0),
                    8)

        # TAMPILKAN STATUS TELAPAK
        cv2.putText(frame, palm_label,
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 0, 0),
                    3)

    cv2.imshow("Deteksi Huruf SIBI Semi Stabil", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()