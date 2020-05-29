import cv2
import os
import time
import json
from utils.utils import read_true_name, load_ground_truth
from src.encodings import FaceEncoder
from src.authenticator import FaceAuthenticator

WINDOW_NAME = "Face Authentication"

PRECOMPUTED_ENCS = True


def test(ref_path, input_path):

    # ---------------------------------------------------------#
    face_encoder = FaceEncoder(PRECOMPUTED_ENCS, input_path)
    face_authenticator = FaceAuthenticator(ref_path)
    # ---------------------------------------------------------#

    while True:

        frame = None
        # run the face tracker
        face_detected, encoding, _ = face_encoder.run(frame)
        # if no encodings are detected use the next frame
        if face_detected:
            # run face authenticator
            stop_recognition, final_answer = face_authenticator.run(encoding)

            # if a final answer is given stop the recognition
            if stop_recognition:
                # True or False on the fact that the person is recognized or not
                result = final_answer
                break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return result


def run():
    enc_path = 'data/encodings/'
    ref_videos = os.listdir(enc_path)
    input_videos = ref_videos
    ground_truth = load_ground_truth()
    total_T = 0
    positive_T = 0
    total_F = 0
    positive_F = 0

    # the print are used to give some insight about performances
    # the print at the bottom give us the time to compare one with all the others
    for ref in ref_videos:

        start = time.time()

        for inputv in input_videos:
            res = test(enc_path + ref, enc_path + inputv)
            ref_w = ref[:-4] + ".webm"
            input_w = inputv[:-4] + ".webm"
            ref_person = read_true_name(ref_w, ground_truth)
            input_person = read_true_name(input_w, ground_truth)
            print("REF: ", ref_person, "GT input: ", input_person, "same person?:", res)
            # TEST T: SAME PERSON
            if ref_person == input_person:
                total_T += 1
                if res:
                    positive_T += 1
            # TEST F: DIFFERENT PERSON
            else:
                total_F += 1
                if res:
                    positive_F += 1
        end = time.time()
        print('Time to comp: ', end - start)

    TAR = positive_T / total_T
    FAR = positive_F / total_F
    print("TAR: ", TAR)
    print("FAR: ", FAR)


run()
