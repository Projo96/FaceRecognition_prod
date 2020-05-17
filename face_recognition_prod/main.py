import cv2

from src.utils import compress
from src.encodings import FaceEncoder
from src.authenticator import FaceAuthenticator

WINDOW_NAME = "FaceRecognition_test"


def run():
    # init video feed
    cv2.namedWindow(WINDOW_NAME)
    # capture the video from the webcam
    video_capture = cv2.VideoCapture(0)
    face_encoder = FaceEncoder()

    path = 'C:/Users/Mattia Proietto/Desktop/PRIM Dataset/encodings_face_videos_half/rec-aae24d0b-11c8-432e-8122-126e4ea60611-RUzrbFxkHM-Qm34q3yjV8r2KLKI-1584629640425-video-face.webm.pkl'  # path of the test encodings
    face_authenticator = FaceAuthenticator(path)
    v_path = 'C:/Users/Mattia Proietto/Desktop/PRIM Dataset/rec-920e9140-85df-4c9f-976a-cb1dbf965631-7avfemevEJ-KugjJX1pMJjZl6hH-1579721759218-video-face.webm'

    v2_path = 'C:/Users/Mattia Proietto/Desktop/PRIM Dataset/rec-b8b6dd4f-d57d-45ad-b0aa-19cd47baa410-DShHqPGa9M-wH081WkwGS0T7bwc-1580309908256-video-face.webm'
    video_capture = cv2.VideoCapture(v2_path)  # video path

    i = 0
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


if __name__ == "__main__":
    run()
