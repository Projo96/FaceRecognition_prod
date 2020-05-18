import cv2
import numpy as np
import face_recognition

# from imutils import face_utils
#import dlib


#from src.config import get_algorithm_params
#from src.utils import find_best_bounding_box

METHOD = 'cnn'

class FaceEncoder:
    
    def __init__(self):
        self.detection = METHOD

    # TODO: function names should be lowercase
    def filterResolutionAndFrontFacing(self, frame, boxes, landmarks):
        # TODO: maybe this function needs to be split, as it does 2 things: it filters on front facing,
        # TODO: but actually not on ratios, it simply computes them
        # defining array where to store face ratio (face_area/image_area)
        ratios = []
        # defining array where to store boxes that where we had front-facing pictures
        newboxes = []
        
        print(len(boxes), len(landmarks))
    
        for i, face_location in enumerate(boxes):

            # TODO: raising errors should be avoided when possible: could you simply check with len(landmarks)?
            # choose the corresponding landmark and skip if the picture was not front-facing
            try:
                # TODO: are you sure that boxes and landmarks are going to be outputed in the same order?
                landmark = landmarks[i]
            except IndexError:
                # if the picture is not front facing the landmark could not exist
                landmark = []
            # TODO you can put that in else statement of the above if else statement
            #check if landmark existed or not
            if(not landmark):
                continue
            # TODO this should be in the if statement
            # extract coordinates from single box
            top, right, bottom, left = face_location

            # compute face area and image area
            area_face = (bottom-top) * (right-left)
            area_image = frame.shape[0] * frame.shape[1]
            ratio = area_face/area_image * 100
    
            # add front-faced picture box and ratio in the new arrays
            ratios.append(ratio)
            newboxes.append(face_location)
            
        return newboxes, ratios
    
    def chooseBestEncoding(self, ratios, encodings):
        feedback = True
        #if there are no face encodings the answer is False
        if(not encodings):
            return False, None
        
        #if there are no face rations were computed the answer is False
        if(not ratios):
            return False, None
       
        # Otherwise, we compute the highest face ration encoding
        main_encoding_index = np.argmax(ratios)
        chosen_encoding = encodings[main_encoding_index]
        
        # The output is returned as a dictionary
        enc_dict = {"encodings": chosen_encoding}
        return feedback, enc_dict

    # TODO: this function could be re-written the following way:
    def chooseBestEncoding_NEW(self, ratios, encodings):

        feedback = False
        enc_dict = {}

        if encodings and ratios:
            # Otherwise, we compute the highest face ration encoding
            main_encoding_index = np.argmax(ratios)
            chosen_encoding = encodings[main_encoding_index]

            # The output is returned as a dictionary
            enc_dict = {"encodings": chosen_encoding}
            feedback = True

            return feedback, enc_dict

        return feedback, enc_dict

    def run(self, frame):
        # convert the frame in the RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # estimate landmark to understand face position
        landmarks = face_recognition.face_landmarks(frame)
        # detect the (x, y)-coordinates of the bounding boxes corresponding to each face
        boxes = face_recognition.face_locations(frame, model=self.detection)
        # measure face area ratio and filter the box if not front-faced
        boxes, ratios = self.filterResolutionAndFrontFacing(frame, boxes, landmarks)
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(frame, boxes)
        # choose the encoding with the highest ratio (main face in the picture)
        feedback, best_encoding = self.chooseBestEncoding(ratios, encodings)
        
        return feedback, best_encoding
    
    

