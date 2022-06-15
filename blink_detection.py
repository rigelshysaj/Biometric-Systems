import cv2
from datetime import datetime
import matplotlib.pyplot as plt


def face_and_eye_detection():
    """Returns images from video feed and recognizes blinks"""
    # Face and eye cascade classifiers from xml files
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    first_read = True
    eye_detected = False
    blink_detected = False
    # Video Capturing by using webcam
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()
    start = datetime.now()

    video_images = []
    while ret:
        # this will keep the webcam running and capturing the image for every loop
        ret, image = cap.read()

        return_image = image.copy()
        video_images += [return_image]
        # Convert the rgb image to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Applying bilateral filters to remove impurities
        gray = cv2.bilateralFilter(gray, 5, 1, 1)
        # to detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (1, 190, 200), 2)
                # face detector
                roi_face = gray[y:y + h, x:x + w]
                # image
                roi_face_clr = image[y:y + h, x:x + w]
                # to detect eyes
                eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_face_clr, (ex, ey), (ex + ew, ey + eh), (255, 153, 255), 2)
                    if len(eyes) >= 2:
                        if first_read:
                            cv2.putText(image, "Eye's detected", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                        1, (255, 0, 0), 2)
                            eye_detected = True
                        else:
                            cv2.putText(image, "Eye's Open", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                        1, (255, 255, 255), 2)
                    else:
                        if first_read:
                            cv2.putText(image, "No Eye's detected", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                        1, (255, 0, 255), 2)
                        else:
                            cv2.putText(image, "Blink Detected.....!!!!", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                                        1, (0, 0, 0), 2)
                            print("Blink Detected.....!!!!")
                            blink_detected = True

        else:
            cv2.putText(image, "No Face Detected.", (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
                        1, (0, 255, 255), 2)
        cv2.imshow('Blink', image)
        cv2.waitKey(1)

        end = datetime.now() - start

        if end.seconds > 5 and eye_detected:
            first_read = False
        if end.seconds > 9 and blink_detected:
            break

    # release the webcam
    cap.release()
    # close the window
    cv2.destroyAllWindows()

    if not blink_detected:
        return "No face or blink detected"

    return video_images, blink_detected


def detect_eyes(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    image = cv2.resize(image, (600, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 1, 1)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    roi_face = gray[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))
    return eyes


def test_accuracy_blink():
    true_positive_rates = []
    false_positive_rates = []
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    for directory in os.listdir("blink"):
        if directory not in ["OpenFace", "ClosedFace"]:
            continue
        for filename in os.listdir("blink/" + directory):
            image = cv2.imread("blink/" + directory + "/" + filename)
            if image is None:
                continue

            eyes = detect_eyes(image)
            if eyes is None:
                continue

            if len(eyes) >= 2:
                if directory == "OpenFace":
                    tp += 1
                else:
                    fp += 1
            else:
                if directory == "ClosedFace":
                    tn += 1
                else:
                    fn += 1

    tpr = tp/(tp + fn)
    fpr = fp/(fp + tn)
    true_positive_rates += [tpr]
    false_positive_rates += [fpr]

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    print("accuracy is: " + str(accuracy))

    print(true_positive_rates)
    print(false_positive_rates)
    print(f"TP {tp}, TN: {tn}, FP {fp}, FN: {fn}")
    data = [[tp, fp], [fn, tn]]
    heatmap = plt.pcolor(data)
    plt.colorbar(heatmap)
    plt.show()


test_accuracy_blink()