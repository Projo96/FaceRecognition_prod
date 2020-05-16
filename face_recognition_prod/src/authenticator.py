import numpy as np
from imutils import face_utils
import dlib
import face_recognition
from src.config import get_algorithm_params
from src.utils import encodingsRead

FACE_AUTHENTICATOR = "FACE_AUTHENTICATOR"


class FaceAuthenticator:

    def __init__(self, encoding_path):

        # load the parameters
        self.params = get_algorithm_params(FACE_AUTHENTICATOR.lower())

        # prepare the output frame:
        self.output_frame = np.zeros((0, 0))

        self.distances = []
        self.analysed_frames = 0
        self.unk_frames = 0
        self.saved_encodings = encodingsRead(encoding_path)


    def run(self, encoding):
        # compute the bounding box

        recognised, dist = self.CompareEncodings(self.saved_encodings, encoding['encodings'])
        feedback, final_decision = self.faceRecOnENC(recognised, dist, len(self.saved_encodings))

        return feedback, final_decision

    def CompareEncodings(self, saved_enc, unk_enc):
        '''
        New version it only works with already made encodings
        - unk_enc: SINGLE encoding of the person we want to recognize
        - saved_enc: dict list of encodings and name of the person of which we know the name-->data on our server
        - tolerance = max eculidean distance between two encoding to be recognised as the same

        Returns:
          name: True if recognized False otherwise
          dist:the min distance between the new face and the one saved if recognized, 1 otherwise
        '''

        # initialize the list of names for each face detected
        dist = 1.0  # distance assigned to non recognized encodings

        # attempt to match each encoding to our known encodings
        # global_number_enc : global parameter that allow us to chose the numeber of known encoding on the server to consider in the comparison
        # if it is <0 all the encodings are taken
        '''
        if global_number_enc > 0:
            data = data["encodings"][0:global_number_enc]
        else:
            data = data["encodings"]
        '''
        ret = False

        data = saved_enc['encodings']
        # Compare a list of face encodings against a candidate encoding to see if they match
        # based on the tolerance parameter
        # Returns:	A list of True/False values indicating which known_face_encodings match the face encoding to check
        matches = face_recognition.compare_faces(data, unk_enc, tolerance=self.params['tolerance'])

        # check to see if we have found a match
        if True in matches:
            # Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
            # for each comparison face.
            # The distance tells you how similar the faces are.
            face_distances = face_recognition.face_distance(data, unk_enc)

            # Choose the known face with the smallest distance to the new face
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                ret = True
                dist = face_distances[best_match_index]

        return ret, dist

    def makeDecision(self, distances, n_frames, mod='avg'):
        '''
        - distances: list of all the face distances so far
        - threshold: percentage of frames to be recognised to say True when distances don't works
        - n_frames: current frame number
        - mod : avg or min , takes the average or the minimum of the distances in order to give the answer
        - Return: True/false to say if the two people are the same
        '''

        if len(distances) > 0:
            if mod == 'avg':
                min_d = np.mean(distances)
            else:
                if mod == 'min':
                    min_d = np.amin(distances)
                else:
                    print('Only avg or min supported the default will be used')
                    min_d = np.mean(distances)

            dists = np.linspace(0.2, 0.45, 50)
            frames = np.linspace(int(self.params['min_frames_to_compare'] / 5),
                                 self.params['min_frames_to_compare'] * 2, 50)
            '''
            HAND-MADE VARIANT to explain how it ideally works
            if min_d <= 0.2:
              return True
            if min_d <= 0.25 and len(distances)>=(2/3)*min_frames_to_compare:
              return True
            if min_d <= 0.3 and len(distances)>=(3/4)*min_frames_to_compare:
              return True
            if min_d <= 0.35 and len(distances)>=(4/5)*min_frames_to_compare:
              return True
            if min_d <= 0.40 and len(distances)>=(5/6)*min_frames_to_compare:
              return True   
            if min_d <= 0.45 and len(distances)>=(6/7)*min_frames_to_compare:
              return True
            if min_d <= 0.50 and len(distances)>=(7/8)*min_frames_to_compare:
              return True
            '''
            # compact version of the above explanation
            for i in range(0, len(dists)):
                if min_d <= dists[i] and len(distances) >= frames[i]:
                    return True

        # we use again a threshold based decision when the previous logig doesn't work
        if len(distances) / n_frames > self.params['threshold']:
            return True

        return False

    def faceRecOnENC(self, recognised, dist, length):

        # we had some cases where no faces where recognised
        if length <= 0:
            print('NO ENCODINGS SAVED , identification is not possible')
            return False, False

        self.analysed_frames += 1

        if not recognised:
            self.unk_frames += 1
        else:
            self.distances.append(dist)

        # -------------------------------------------------------------------------------------------------------------#

        # early stopping for negative result
        # After tot frames we try to see if we can give a negative answer before the end of the video
        # in particular we check if more than half of the frames weren't
        # recognised at all and if that is the case we can say that
        # the two person are different
        if self.analysed_frames >= self.params['min_frames_to_compare']:  # 45 frames = 1.5 second  -- early stopping

            if self.unk_frames / self.analysed_frames > 0.5:  # early stopping for the negative
                return True, False

        # early stopping for positive result
        # starting from min_frames/5 every time we try to give an answer using the function make decision
        if self.analysed_frames >= self.params[
            'min_frames_to_compare'] / 5:  # ex 30fps --> 15 frames = 0.5 second  ---early stopping

            if self.makeDecision(self.distances, self.analysed_frames, mod=self.params['mod']):
                return True, True

        # -------------------------------------------------------------------------------------------------------------#

        # for ends
        if self.analysed_frames == 0:
            self.analysed_frames = 1

        if self.analysed_frames < self.params['min_frames_to_compare'] * 2:
            return False, False

        # ex 85% of the frames are recognised
        if (self.analysed_frames - self.unk_frames) / self.analysed_frames > self.params['threshold']:
            return True, True
        else:
            return True, False
