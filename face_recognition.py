import cv2
import os
from pathlib import Path
import numpy as np
from random import sample
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
FACES_TEST = "faces_test.npy"
FACES_TRAIN = "faces_train.npy"
LABELS_TEST = "labels_test.npy"
LABELS_TRAIN = "labels_train.npy"


def detect_face(image):
    """Returns a tuple containing (image cropped to face, the matrix representation of it)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]


def prepare_data_for_training(limit):
    """Returns a tuple containing (faces_train, labels_train, faces_test, labels_test)
    only performs face detection from images if a previous result is not already saved to disk"""
    names_dir = Path.joinpath(ROOT, "faces/lfw-deepfunneled/")
    extracted_faces_dir = "extracted_faces_data/"

    if FACES_TEST not in os.listdir(Path.joinpath(ROOT, extracted_faces_dir)):
        faces = []
        labels = []
        count = 0
        limit_count = 0
        for name in os.listdir(names_dir):
            count += 1
            for image in os.listdir(Path.joinpath(names_dir, name)):
                if len(os.listdir(Path.joinpath(names_dir, name))) > 2:
                    limit_count += 1
                    if limit_count >= limit:
                        break
                    filename = Path.joinpath(names_dir, name, image)
                    im = cv2.imread(str(filename))
                    face, rect = detect_face(im)
                    if face is not None:
                        faces += [face]
                        labels += [count]
                        print(f"{limit_count}. Elaborato {image} per tizio {name}")
            if limit_count >= limit:
                break

        faces_train = []
        faces_test = []
        labels_train = []
        labels_test = []

        indices = [x for x in range(len(faces))]
        test_indices = sample(indices, len(faces) // 10)
        for i in indices:
            if i in test_indices:
                faces_test += [faces[i]]
                labels_test += [labels[i]]
            else:
                faces_train += [faces[i]]
                labels_train += [labels[i]]

        np.save(extracted_faces_dir + FACES_TEST, faces_test)
        np.save(extracted_faces_dir + FACES_TRAIN, faces_train)
        np.save(extracted_faces_dir + LABELS_TEST, labels_test)
        np.save(extracted_faces_dir + LABELS_TRAIN, labels_train)
    else:
        faces_test = np.load(extracted_faces_dir + FACES_TEST, allow_pickle=True)
        faces_train = np.load(extracted_faces_dir + FACES_TRAIN, allow_pickle=True)
        labels_test = np.load(extracted_faces_dir + LABELS_TEST, allow_pickle=True)
        labels_train = np.load(extracted_faces_dir + LABELS_TRAIN, allow_pickle=True)
        print("Loaded train and test data from disk.")

    return faces_train, labels_train, faces_test, labels_test


def extract_faces_and_labels(dataset_dir):
    """Returns two lists: (faces, labels) starting from images in the dataset directory"""
    faces = []
    labels = []

    count = 0
    for name in os.listdir(dataset_dir):
        count += 1
        for image in os.listdir(Path.joinpath(dataset_dir, name)):
            filename = Path.joinpath(dataset_dir, name, image)
            im = cv2.imread(str(filename))
            face, rect = detect_face(im)
            if face is not None:
                faces += [face]
                labels += [count]
                print(f"{count}. Elaborato {image} per tizio {name}")
    return faces, labels


def prepare_data_for_training_celeb():
    train_dir = Path("celeb_faces/train")
    test_dir = Path("celeb_faces/test")

    faces_train, labels_train = extract_faces_and_labels(train_dir)
    faces_test, labels_test = extract_faces_and_labels(test_dir)
    return faces_train, labels_train, faces_test, labels_test


def train_models(faces, labels, force_lbph=False, force_eigen=False, force_fisher=False, do_lbph=True, do_eigen=False, do_fisher=False):
    labels = np.array(labels)

    if "models" not in os.listdir():
        os.mkdir("models")

    lbph_face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=3)
    if do_lbph:
        if "LBPH.yml" not in os.listdir("models") or force_lbph:
            lbph_face_recognizer.train(faces, labels)
            lbph_face_recognizer.save("models/LBPH.yml")
            print("Finished LBPH and saved to models/LBPH.yml")
        else:
            lbph_face_recognizer.read("models/LBPH.yml")
            print("LBPH face recognizer loaded from pretrained model")

    eigen_face_recognizer = cv2.face.EigenFaceRecognizer_create()
    if do_eigen:
        if "Eigen.yml" not in os.listdir("models") or force_eigen:
            eigen_face_recognizer.train(faces, labels)
            eigen_face_recognizer.save("models/Eigen.yml")
            print("Finished Eigen and saved to models/Eigen.yml")
        else:
            eigen_face_recognizer.read("models/Eigen.yml")
            print("Eigen face recognizer loaded from pretrained model")

    fisher_face_recognizer = cv2.face.FisherFaceRecognizer_create()
    if do_fisher:
        if "Fisher.yml" not in os.listdir("models") or force_fisher:
            fisher_face_recognizer.train(faces, labels)
            fisher_face_recognizer.save("models/Fisher.yml")
            print("Finished Fisher and saved to models/Fisher.yml")
        else:
            fisher_face_recognizer.read("models/Eigen.yml")
            print("Fisher face recognizer loaded from pretrained model")

    return lbph_face_recognizer, eigen_face_recognizer, fisher_face_recognizer


def predict(face_recognizer, test_image):
    label, confidence = face_recognizer.predict(test_image)
    return label, confidence


def get_names(directory):
    if directory == "normal":
        names_dir = Path.joinpath(ROOT, "faces/lfw-deepfunneled/")
    if directory == "celeb":
        names_dir = Path.joinpath(ROOT, "celeb_faces/test/")

    names = [x.name for x in names_dir.iterdir()]
    return names


def test_accuracy(face_recognizer, faces_test, labels_test, directory):
    correct = 0
    total = len(faces_test)
    names = get_names(directory)
    for index, face in enumerate(faces_test):
        guess, distance = predict(face_recognizer, face)
        if guess == labels_test[index]:
            correct += 1
            print(
                f"Correct guess for {names[guess - 1]}. Accuracy: {correct / (index + 1)}, correct guesses: {correct}, total: {index + 1}. Distance: {distance}")
        else:
            print(
                f"Wrong   guess for {names[labels_test[index] - 1]}. Accuracy: {correct / (index + 1)}, correct guesses: {correct}, total: {index + 1}. Guess was: {names[guess - 1]}. Distance: {distance}")
    print(f"Accuracy: {correct / total}, correct guesses: {correct}, total: {total}")


def test_accuracy_thresholds(face_recognizer, faces_test, labels_test, directory):
    statements = []
    thresholds = []
    false_accept_values = []
    false_reject_values = []
    true_accept_values = []
    true_reject_values = []
    accuracies = []
    for threshold in range(1, 201, 10):
        true_accept = 0
        false_accept = 0
        true_reject = 0
        false_reject = 0
        total = len(faces_test)
        names = get_names(directory)
        correct = 0
        for index, face in enumerate(faces_test):
            guess, distance = predict(face_recognizer, face)
            if names[labels_test[index] - 1] != "ZZZ":
                # Here we can have true accept or false reject
                if distance > threshold:
                    false_reject += 1
                else:
                    true_accept += 1
            else:
                # Here we can have false accept or true reject
                if distance > threshold:
                    true_reject += 1
                else:
                    false_accept += 1
        statements += [f"With threshold: {threshold}: TR: {true_reject}, TA: {true_accept}, FR: {false_reject}, FA: {false_accept}, accuracy: {(true_accept + true_reject) / total}"]
        thresholds += [threshold]
        true_reject_values += [true_reject]
        true_accept_values += [true_accept]
        false_reject_values += [false_reject]
        false_accept_values += [false_accept]
        accuracies += [(true_accept + true_reject) / total]
    for statement in statements:
        print(statement)

    #TODO: fai tutti i plot più o meno allo stesso modo
    plt.plot(thresholds, accuracies)
    plt.show()
