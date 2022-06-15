import cv2
from face_recognition import train_models, test_accuracy, prepare_data_for_training, prepare_data_for_training_celeb, test_accuracy_thresholds


def show_image(image):
    cv2.imshow("Training on image...", image)
    cv2.waitKey(2000)


def reshape(faces):
    new_faces = []
    for f in faces:
        x = f.copy()
        x = cv2.resize(x, (100, 100))
        new_faces += [x]
    return new_faces


if __name__ == '__main__':
    faces_train, labels_train, faces_test, labels_test = prepare_data_for_training_celeb()

    faces_train = reshape(faces_train)
    faces_test = reshape(faces_test)
    print("Reshape finished")


    # fai train di tutti e 3 i modelli e salva su disk

    LBPH_recognizer, Eigen_recognizer, Fisher_recognizer = train_models(faces_train, labels_train)
    print("Training finished")

    #TODO: aggiungi qui cose per fare confronti tra i vari modelli
    test_accuracy_thresholds(LBPH_recognizer, faces_test, labels_test, "celeb")
