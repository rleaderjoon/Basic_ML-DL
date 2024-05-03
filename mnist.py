
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('accuracy') > 0.95):
            print("\n정확도 95%에 도달하여 훈련을 멈춥니다.")
            self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    # Flatten은 층이 아니라 입력을 위한 크기를 지정
    # 2D 배열인 행렬을 1D 배열인 벡터로 변환
    tf.keras.layers.Flatten(input_shape = (28,28)),
    # Hidden Layer
    # 입출력 사이에 위치해 보이지 않기 때문
    # 뉴런의 개수는 Hyperparameter
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    # 출력 층
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs = 50, callbacks = [callbacks])