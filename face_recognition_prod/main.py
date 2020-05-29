import cv2

from utils.utils import compress
from src.encodings import FaceEncoder
from src.authenticator import FaceAuthenticator

WINDOW_NAME = "Face Authentication"

PRECOMPUTED_ENCS = True
DEMO_MODE = 1
NO_DEMO = 0
REFERENCE_NAME = "rec-e3f2bdb0-1ca5-4b0f-9dad-5e50b64eb363-xfevbjX3K7-s6B79p8qv6E5X1mA-1581874242208-video-face"
FILE_NAME = "rec-0a4f48de-9fec-453a-bca4-83aae95b24bf-1DsGU3mckh-SBI6aoDfEp5GzV46-1582013079945-video-face"


def run():
    # If you want some intermediate results to be displayed on screen, set demo_mode to 1
    demo_mode = DEMO_MODE

    # init video feed
    # cv2.namedWindow(WINDOW_NAME)
    # capture the video from the web-cam
    video_path = "data/face_videos/" + FILE_NAME + ".webm"
    video_capture = cv2.VideoCapture(video_path)

    # ---------------------------------------------------------#
    ref_path = "data/encodings/" + REFERENCE_NAME + ".pkl"
    input_path = "data/encodings/" + FILE_NAME + ".pkl"

    face_encoder = FaceEncoder(PRECOMPUTED_ENCS, input_path, demo_mode=demo_mode)
    face_authenticator = FaceAuthenticator(ref_path, demo_mode=demo_mode)
    # ---------------------------------------------------------#

    while True:
        if not PRECOMPUTED_ENCS:
            ret, frame = video_capture.read(0)
            frame = compress(frame, 2)  # to make it run faster
        else:
            ret = True
            frame = None

        if not ret:
            break

        # run the face tracker
        face_detected, encoding, output_frame = face_encoder.run(frame)
        print("         face detected: ", face_detected)
        # if no encodings are detected use the next frame
        if face_detected:
            # run face authenticator
            stop_recognition, final_answer = face_authenticator.run(encoding)

            # if a final answer is given stop the recognition
            if stop_recognition:
                # True or False on the fact that the person is recognized or not
                print("USER IS THE SAME AS REFERENCE: ", final_answer)
                break

        # show frames
        if not PRECOMPUTED_ENCS:
            cv2.imshow(WINDOW_NAME, output_frame)
            cv2.moveWindow(WINDOW_NAME, -200, -800)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
