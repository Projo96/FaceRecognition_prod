import cv2

from src.utils import compress
from src.encodings import FaceEncoder
from src.authenticator import FaceAuthenticator

WINDOW_NAME = "FaceAthentication"


def run():
    # init video feed
    cv2.namedWindow(WINDOW_NAME)
    # capture the video from the webcam
    video_capture = cv2.VideoCapture(0)
    face_encoder = FaceEncoder()

    path = 'C:/Users/super/Jupypter/#11/encodings_88/fa.pkl'  # path of the test encodings
    face_authenticator = FaceAuthenticator(path)

    while True:
        ret, frame = video_capture.read()
        frame = compress(frame, 2)  # to make it run faster

        if not ret:
            break

        # run the face tracker
        feedback, encoding = face_encoder.run(frame)
        # if no encodings are detected use the next frame
        if not feedback:
            continue
        # run face authenticator
        feedback, response = face_authenticator.run(encoding)

        # if a final answer is given stop the recognition
        if feedback:
            print(response)
            break

        # show frames
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
     
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
