from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

#=============================
import tensorflow as tf
import random as rn
import numpy as np
seed_num = 1 # seed_num은 1로 고정(변경하지 말 것) 
np.random.seed(seed_num)
rn.seed(seed_num)
tf.random.set_seed(seed_num)
#=============================


# 데이터 입력
df = pd.read_csv('./data_grade.csv', names = ["sudent", "paran ts", "hand", "contents", "class", "notice", "discuss", "survey", "satisfaction", "absent"])
#print(df.head())


# 데이터 분류
dataset = df.values
#X = dataset[:,0:4].astype(float)
X = dataset[:,[3,5,6,9]].astype(float)/100 # 속성
X[:,3] = X[:,3]*100

Y_obj = dataset[:,4] # 정답 클래스

# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = to_categorical(Y)


# 학습셋과 테스트셋의 구분
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.25, shuffle=True, random_state=seed_num)

model = Sequential()
# 모델의 설정
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
for i in range(17):
    model.add(Dense(10, activation= 'relu'))
model.add(Dense(3, activation= 'softmax'))
# 모델 컴파일
model.compile(loss= 'mean_squared_error',
              optimizer= 'adam',
              metrics=['accuracy'])
# 모델 실행
history = model.fit(X_train, Y_train, epochs=100, batch_size=10)
print("\n\n")
# 결과 출력
print("학습데이터의 정확도: %.4f \n\n" % (model.evaluate(X_train, Y_train)[1]))
print("테스트셋의 정확도: %.4f"% (model.evaluate(X_test, Y_test)[1]))

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['loss'], label = 'loss')
plt.legend()
plt.show()