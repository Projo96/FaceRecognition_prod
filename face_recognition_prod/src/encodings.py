import cv2
import numpy as np
import face_recognition
from utils.utils import encodings_read, assign_landmark_to_box, find_biggest_box, plot_feedback

METHOD = 'cnn'


class FaceEncoder:

    def __init__(self, precomputed, input_path, demo_mode=0):
        # defines the method used
        self.detection = METHOD
        # this is true is encodings already exist
        self.precomputed_encs = precomputed
        self.demo_mode = demo_mode
        if self.precomputed_encs:
            # initialise the counter
            self.counter = 0
            # set the path where to take the encoding
            self.input_path = input_path
            # reads the encoding
            self.encodings = encodings_read(self.input_path)

    @staticmethod
    def filter_front_facing(boxes, landmarks):
        """
        If a box has no landmarks, it isn't considered as front facing
        """
        # defining array where to store boxes where we had front-facing pictures
        new_boxes = []

        # landmark and box are independent each other
        for landmark in landmarks:
            # check the compatibility between the box and the landmark (sometimes it can happen that landmarks don't
            # come out in the same order than the boxes, so we need to match them)
            box = assign_landmark_to_box(landmark, boxes)
            # every confirmed box is added to the new_boxes array
            if box is not None:
                new_boxes.append(box)

        return new_boxes

    def filter_found_faces(self, frame, boxes, landmarks):
        # initialise result array
        main_box = []

        if boxes and landmarks:
            # finds the biggest face box in the picture
            main_box = find_biggest_box(frame, boxes)
            # matches the given face box with the landmarks detected
            main_box = self.filter_front_facing(main_box, landmarks)

        # returns the main face box in the frame or an empty list
        return main_box

    @staticmethod
    def create_and_format_encoding(frame, main_box):
        # initialise the false feedback
        feedback = False

        if main_box:
            # if there is a box and a related landmark we will have an encoding
            feedback = True
            # takes the first encoding from the list (we always have one box here)
            encoding = face_recognition.face_encodings(frame, main_box)[0]
            # encoding is formatted as a dictionary
            enc_dict = {"encodings": encoding}
            # returns the feedback and the dictionary
            return feedback, enc_dict

        return feedback, None

    def run(self, frame):
        # if we are working with already create encodings
        if self.precomputed_encs:
            feedback, best_encoding = self.encoding_fetcher()

        else:
            # convert the frame in the RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # estimate landmark to understand face position
            landmarks = face_recognition.face_landmarks(frame)
            # detect the (x, y)-coordinates of the bounding boxes corresponding to each face
            boxes = face_recognition.face_locations(frame, model=self.detection)
            # look for main face in the picture that is front-facing
            main_box = self.filter_found_faces(frame, boxes, landmarks)
            # compute the facial embedding on the main face box found
            feedback, best_encoding = self.create_and_format_encoding(frame, main_box)

            if self.demo_mode > 0:
                frame = plot_feedback(frame, landmarks, boxes, main_box)

        return feedback, best_encoding, frame

    def encoding_fetcher(self):
        # initialise return values
        feedback = False
        enc_dict = None

        # if there are still enough encodings
        if self.counter < len(self.encodings['encodings']):
            feedback = True
            # takes the already filtered encoding and increases the counter
            encoding = self.encodings['encodings'][self.counter]
            self.counter += 1
            # encoding is formatted as a dictionary
            enc_dict = {"encodings": encoding}

        return feedback, enc_dict
