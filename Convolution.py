import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

dense = Dense(units = 1, input_shape=[1,])

# 신경망 정의
# units = 1이기 때문에 1개의 뉴런을 가집니다.
# 입력 데이터가 X이고 숫자 하나이므로 [1]
model = Sequential([dense])

# 추측이 얼마나 좋고 나쁜지를 측정하는 Loss Function
# Loss Function을 바탕으로 다시 추측을 시작하는 것이 Optimizer
# SGD = Stochastic Gradient Descent
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

# Tensorflow는 Numpy를 이용함
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# xs배열과 ys배열을 이용해여 epochs만큼 훈련하라
model.fit(xs, ys, epochs = 500)

print(model.predict(np.array([10.0])))
print("신경망이 학습한 것 : {}".format(dense.get_weights()))