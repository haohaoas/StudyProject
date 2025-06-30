import random
import cv2
from keras import datasets, models, layers
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28 * 28))
x_test = x_test.reshape((10000, 28 * 28))
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
model=models.Sequential(
    [
        layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_acc:', test_acc)

index = random.randint(0,11)
image = x_test[index]
label = y_test[index]
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"MNIST 测试集中第 {index} 张图像，标签是 {label}")
plt.axis('off')
plt.show()
img_to_predict = np.reshape(image, (1, 28 * 28))
predictions = model.predict(img_to_predict)
predicted_class = np.argmax(predictions[0])

print(f"真实标签是: { label}")
print(f"模型预测的数字是: {predicted_class}")
print(f"所有数字的预测概率是: {predictions[0]}")
