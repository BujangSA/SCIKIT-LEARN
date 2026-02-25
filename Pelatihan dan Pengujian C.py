import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


with open("data.pickle", "rb") as f:
    data_dict = pickle.load(f)

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

print("Jumlah data:", len(data))

if len(data) == 0:
    print("Dataset kosong! Jalankan ekstraksi fitur dulu.")
    exit()


x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.1,
    shuffle=True,
    stratify=labels
)


model = RandomForestClassifier()
model.fit(x_train, y_train)


y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f'{score * 100:.2f}% of samples were classified correctly!')


with open("model.pickle", "wb") as f:
    pickle.dump(model, f)

print("Model berhasil disimpan ke model.pickle")