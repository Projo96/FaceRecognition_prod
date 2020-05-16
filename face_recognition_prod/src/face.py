import cv2
import numpy as np
from imutils import face_utils
import dlib

from src.config import get_algorithm_params
from src.utils import find_best_bounding_box


NO_FACE_IN_FRAME = "NO_FACE_IN_FRAME"
FACE_DETECTED = "FACE_DETECTED"
FACE_TRACKER = "FACE_TRACKER"


class FaceTracker:

    def __init__(self):

        # load the parameters
        self.params = get_algorithm_params(FACE_TRACKER.lower())

        # prepare the output frame:
        self.output_frame = np.zeros((0, 0))

        # load dlib detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        #self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    def run(self, frame):
        # compute the bounding box
        # this algorithm requires gray scale frames
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # set the frame which will be used for visualization
        output_frame = frame.copy()

        # look for a face in the frame, and compute the bounding box
        feedback, bounding_box, output_frame = self.compute_bounding_box(gray_frame, output_frame)

        # if there is a face in the frame, compute the landmarks.
        if feedback == FACE_DETECTED:
            landmarks, output_frame = self.compute_landmarks(gray_frame, bounding_box, output_frame)

        return feedback, output_frame

    def compute_bounding_box(self, gray_frame, output_frame):
        """
        computes the bounding box for an image
        (4 corners in which the face is in)
        :return:
        """
        feedback = NO_FACE_IN_FRAME
        best_bounding_box = []
        # The second argument is the number of times we will upscale the image (in this case we don't, as
        # it increase computation time)
        # The third argument to run is an optional adjustment to the detection threshold,
        # where a negative value will return more detections and a positive value fewer.
        candidate_bounding_boxes, scores, idx = self.detector.run(gray_frame, 1, -0.3)

        if len(candidate_bounding_boxes) > 0:
            feedback = FACE_DETECTED

            # find best bounding box:
            best_bounding_box = find_best_bounding_box(candidate_bounding_boxes, gray_frame)

            output_frame = cv2.rectangle(
                output_frame,
                (best_bounding_box.left(), best_bounding_box.top()),
                (best_bounding_box.right(), best_bounding_box.bottom()),
                (0, 255, 0),
                1,
            )

        # write on the output frame
        color = (0, 255, 0) if feedback == FACE_DETECTED else (0, 0, 255)
        output_frame = cv2.putText(
            output_frame,
            feedback,
            (gray_frame.shape[1] // 2, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

        return feedback, best_bounding_box, output_frame

    def compute_landmarks(self, gray_frame, bounding_box, output_frame):
        """
        The landmarks represents the important points in the face
        We will use them to analyse the evolution of the face towards time
        :return:
        """

        # find landmarks:
        landmarks = self.predictor(gray_frame, bounding_box)

        # convert to numpy array
        landmarks = face_utils.shape_to_np(landmarks)

        output_frame = cv2.rectangle(
            output_frame,
            (bounding_box.left(), bounding_box.top()),
            (bounding_box.right(), bounding_box.bottom()),
            (0, 255, 0),
            1,
        )
        for i, (x, y) in enumerate(landmarks):
            output_frame = cv2.circle(
                output_frame, (x, y), 3, (0, 255, 0), -1
            )

        return landmarks, output_frame

