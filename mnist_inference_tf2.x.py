'''This is inference code for mnist dataset'''

from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import pandas as pd

# 모델 로딩 (compile=False 로 optimizer 제외)
model = load_model("saved_model/", compile=False)

# Show the model architecture
model.summary()

##-- Model Test using Test datasets
print()
print("----Actual test for digits----")

mnist_label_file_path =  "dataset_test/testlabels/t_labels.txt"
mnist_label = open(mnist_label_file_path, "r")
cnt_correct = 0

for index in range(10):
    #-- read a label
    label = mnist_label.readline().strip()

    #-- formatting the input image (image data)
    img = Image.open(f'dataset_test/testimgs/{index+1}.png').convert("L")
    img = img.resize((28, 28))
    im2arr = np.array(img).reshape(1, 28, 28, 1) / 255.0  # normalize

    # Predicting the Test set results
    y_pred = model.predict(im2arr)
    pred_label = np.argmax(y_pred)

    print()
    print(f"label = {label} --> predicted label = {pred_label}")

    #-- compute the accuracy of the prediction
    if int(label) == pred_label:
        cnt_correct += 1

#-- Final accuracy
Final_acc = cnt_correct / 10
print()
print(f"Final test accuracy: {Final_acc:.2f}")
print()
print('****tensorflow version****:', tf.__version__)
print()

# 학과 정보 출력
data = {
    '이름': ['정은서'],
    '학번': [2310093],
    '학과': ['인공지능공학부']
}
df = pd.DataFrame(data)
print(df)
