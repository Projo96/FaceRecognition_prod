import cv2


from src.utils import compress


WINDOW_NAME = "Ubble Interview"


def run():
    # init video feed
    cv2.namedWindow(WINDOW_NAME)
    # capture the video from the webcam
    path = 'ciao'
    video_capture = cv2.VideoCapture(0)#video path


    while True:
        ret, frame = video_capture.read()
        frame = compress(frame, 2)  # to make it run faster

        if not ret:
            break

        # run the face tracker
        #feedback, output_frame = face_tracker.run(frame)

        # show frames
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    run()
