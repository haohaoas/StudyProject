import os
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import Subset

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
def create_dataset(sample_ratio=1.0):
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465],
                  std=[0.2023, 0.1994, 0.2010])
    ])
    train_data = CIFAR10(root='data', train=True, download=False, transform=transform)
    test_data = CIFAR10(root='data', train=False, download=False, transform=transform)

    total_len = len(train_data)
    sample_size = int(total_len * sample_ratio)
    indices = random.sample(range(total_len), sample_size)
    train_subset = Subset(train_data, indices)

    return train_subset, test_data

train_subset, test_data = create_dataset(sample_ratio=1.0)

x_train = np.stack([np.transpose(data[0].numpy(), (1, 2, 0)) for data in train_subset])
y_train = np.array([data[1] for data in train_subset])
x_test = np.stack([np.transpose(data[0].numpy(), (1, 2, 0)) for data in test_data])
y_test = np.array([data[1] for data in test_data])

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ✅ 构建 tf.data.Dataset + 动态 resize
IMG_SIZE = 64
BATCH_SIZE = 32

def preprocess(x, y):
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    return x, y

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
    .cache() \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
    .cache() \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

# ✅ 构建 MobileNetV2 模型
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)

# 解冻后 30 层用于微调
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model.output
predictions = Dense(10, activation='softmax',dtype='float32')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ 自动提前停止
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# ✅ 开始训练
model.fit(train_ds,
          validation_data=test_ds,
          epochs=10,
          callbacks=[early_stop])

# ✅ 最终测试
loss, acc = model.evaluate(test_ds)
print(f"✅ 测试集准确率：{acc:.2%}")