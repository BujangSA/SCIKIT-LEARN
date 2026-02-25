import os
import cv2

DATA_DIR = r"C:\Users\d_rad\PyCharmMiscProject\.venv\Lib\site-packages\sklearn\datasets\tests\dataset"

os.makedirs(DATA_DIR, exist_ok=True)

number_of_classes = 2
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):

    class_path = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_path, exist_ok=True)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera tidak terbaca")
            break

        cv2.putText(frame, 'Ready? Press "Q" !',
                    (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    3)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)
        cv2.waitKey(10)

        cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()