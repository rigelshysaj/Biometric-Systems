import os

import numpy as np
import pandas as pd
import cv2
import random
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import load_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

traindir = "gender/Training"
validdir = "gender/Validation"

if "gendermodel" not in os.listdir("gender"):
    height = 150
    width = 150
    train_datagen = ImageDataGenerator(rescale=1/255.0, rotation_range=45, height_shift_range=0.2, shear_range=0.2,
                                       zoom_range=0.2, validation_split=0.2, horizontal_flip=True)

    train_data = train_datagen.flow_from_directory(directory=traindir, target_size=(height, width),
                                                   class_mode="categorical", batch_size=32, subset="training")

    val_datagen = ImageDataGenerator(rescale=1/255.0)

    val_data = train_datagen.flow_from_directory(directory=traindir, target_size=(height, width),
                                                 class_mode="categorical", batch_size=32, subset="validation")

    #############
    mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_shape=(height, width, 3))
    for layer in mobilenet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(mobilenet)
    model.add(Dense(128, activation="relu"))

    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    checkpoint = ModelCheckpoint("Gender.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

    batch_size = 32
    history = model.fit_generator(train_data, steps_per_epoch=len(train_data) // batch_size, epochs=15,
                                  validation_data=val_data, validation_steps=len(val_data)//batch_size,
                                  callbacks=[checkpoint, earlystop], verbose=1)

    model.evaluate_generator(val_data)
    model.save("gender/gendermodel")
else:
    model = load_model("gender/gendermodel")


def checking(img):
    label = {0: "female", 1: "male"}
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 7)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]

    face = image[y:y + h, x:x + w]
    face = cv2.resize(face, (150, 150))
    img_scaled = face / 255.0
    reshape = np.reshape(img_scaled, (1, 150, 150, 3))
    img = np.vstack([reshape])
    res = model.predict(img)

    return res, img


def test_accuracy_gender():
    label = {0: "female", 1: "male"}
    reverse_label = {"female": 0, "male": 1}
    total = 0
    correct = 0
    for directory in os.listdir("gender/validation"):
        for image in os.listdir("gender/validation/" + directory):
            result = checking("gender/validation/" + directory + "/" + image)
            if result is not None:
                total += 1
                if result == reverse_label[directory]:
                    correct += 1
                print(f"Checking in directory {directory}, predicted {reverse_label[directory]}, correct: {correct}, total: {total}, accuracy: {correct / total}")

    print(f"Accuracy: {correct / total}")


def gender_plot_auroc():
    true_positive_rates = []
    false_positive_rates = []
    reverse_label = {"female": 0, "male": 1}

    thresholds = []
    true_positive_values = []
    false_negative_values = []
    true_negative_values = []
    false_positive_values = []
    accuracies = []
    for i in range(1, 11):
        tn = 0
        tp = 0
        fp = 0
        fn = 0
        for directory in os.listdir("gender/Validation"):
            for image in os.listdir("gender/Validation/" + directory):
                res, img = checking("gender/Validation/" + directory + "/" + image)
                if res and img:
                    result = res[0][0]
                    if result >= i / 10:
                        if 0 == reverse_label[directory]:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if 0 == reverse_label[directory]:
                            fn += 1
                        else:
                            tn += 1
            print(f"Accuracy at threshold {i / 10}: {(tp + tn) / (tp + tn + fp + fn)}")
        tpr = tp/(tp + fn)
        fpr = fp/(fp + tn)
        true_positive_rates += [tpr]
        false_positive_rates += [fpr]
        thresholds += [i]
        true_positive_values += [tp]
        true_negative_values += [tn]
        false_positive_values += [fp]
        false_negative_values += [fn]
        accuracies += [(tp + tn) / (tp + tn + fp + fn)]

    print(f"TP {tp}, TN: {tn}, FP {fp}, FN: {fn}")
    plt.plot(false_positive_rates, true_positive_rates)
    plt.show()
