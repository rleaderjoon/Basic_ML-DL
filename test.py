import tensorflow as tf

# 예시 데이터셋
data = tf.data.Dataset.range(1000)  # 1000개의 예시 데이터를 가진 데이터셋 생성

# 셔플링과 배칭
train_batches = data.shuffle(100).batch(10)

# 배치 데이터를 출력해보는 예시
for batch in train_batches.take(1):
    print(batch.numpy())
