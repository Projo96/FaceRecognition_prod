import cv2

from src.utils import compress
from src.encodings import FaceEncoder
from src.authenticator import FaceAuthenticator

WINDOW_NAME = "Face Authentication"


def run():
    # init video feed
    cv2.namedWindow(WINDOW_NAME)
    # capture the video from the web-cam
    video_capture = cv2.VideoCapture(0)

    # or like this if we want use a saved video
    # video_capture = cv2.VideoCapture(video path)
    face_encoder = FaceEncoder()
    #---------------------------------------------------------#
    #!!!!!!!!!!!!!!!!!!!CHANGE THE PATH!!!!!!!!!!!!!!!!!!!!!!!!!

    # TODO: why are we still referring to set A?
    path = 'path of the encodings of the set A'
    face_authenticator = FaceAuthenticator(path)
    #---------------------------------------------------------#

    while True:
        ret, frame = video_capture.read()
        frame = compress(frame, 2)  # to make it run faster

        if not ret:
            break

        # run the face tracker
        feedback, encoding = face_encoder.run(frame)
        # if no encodings are detected use the next frame
        # TODO: you can be more precise with the statement by only allowing to run face_authenticator if feedback
        # has certain value
        if not feedback:
            continue
        # run face authenticator
        # TODO: try to find more explicit names for feedback and response, like this it is hard to understand which does what
        feedback, response = face_authenticator.run(encoding)

        # if a final answer is given stop the recognition
        # TODO: a bit misleading that feedback is also the output of face_encoder.
        if feedback:
            print(response)#True or False based on the fact that the person is recognized or not
            break

        # show frames
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
