import pickle
import cv2
import mediapipe as mp
import numpy as np

with open("model.pickle", "rb") as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

# ==========================
# HANDS
# ==========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# ==========================
# POSE
# ==========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

labels_dict = {
0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',
8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',
20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'
}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ==========================
    # PROCESS HANDS
    # ==========================
    results_hands = hands.process(frame_rgb)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        prediction = model.predict([np.array(data_aux)])
        huruf = labels_dict[int(prediction[0])]

        cv2.putText(frame, f"Huruf: {huruf}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2)

    # ==========================
    # PROCESS POSE
    # ==========================
    results_pose = pose.process(frame_rgb)

    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results_pose.pose_landmarks.landmark

        # Ambil hip & knee kanan
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

        distance = abs(hip.y - knee.y)

        if distance > 0.25:
            posisi = "BERDIRI"
            color = (0, 255, 0)
        else:
            posisi = "DUDUK"
            color = (0, 0, 255)

        cv2.putText(frame, posisi,
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3)

    cv2.imshow("SIBI + Postur Tubuh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()