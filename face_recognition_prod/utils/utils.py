import dlib
import numpy as np
import cv2
import pickle
import json


def plot_feedback(frame, landmarks_for_all_face, boxes, main_box):

    output_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

    for landmarks_for_one_face in landmarks_for_all_face:
        for face_attribute, list_of_landmarks in landmarks_for_one_face.items():
            for (x, y) in list_of_landmarks:
                output_frame = cv2.circle(
                    output_frame, (x, y), 3, (0, 255, 0), -1
                )
    for (top, right, bottom, left) in boxes:
        output_frame = cv2.rectangle(
            output_frame,
            (left, top),
            (right, bottom),
            (255, 0, 0),
            1,
        )

    for (top, right, bottom, left) in main_box:
        output_frame = cv2.rectangle(
            output_frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            1,
        )

    return output_frame


def find_biggest_box(frame, boxes):
    # define ratios array
    ratios = []
    # define new box array
    main_box_list = []

    # compute frame area
    area_image = frame.shape[0] * frame.shape[1]

    for face_location in boxes:
        # extract coordinates from the box
        top, right, bottom, left = face_location
        # compute face area
        area_face = (bottom - top) * (right - left)
        # compute ratio wrt the frame
        ratio = area_face / area_image * 100
        # add value to the array
        ratios.append(ratio)

    # selection of the biggest face in the picture
    if ratios:
        # select the biggest ratio (biggest face)
        main_box_index = np.argmax(ratios)
        # select the corresponding box
        main_box = boxes[main_box_index]
        # append this box to the new box list
        main_box_list.append(main_box)

    return main_box_list


def assign_landmark_to_box(landmark, boxes):
    """
    Matches the box and landmark, analysing if
    the key face features are all positioned inside the box
    """
    landmark_avg = []

    # Taking important key-feature in the person's face
    # We chose these ones because they represent upper, center and lower part
    nose_bridge = landmark['nose_bridge']
    bottom_lip = landmark['bottom_lip']
    right_eyebrow = landmark['right_eyebrow']

    # Computing the average of taken points relative to the part of the face
    nose_bridge = [sum(y) / len(y) for y in zip(*nose_bridge)]
    bottom_lip = [sum(y) / len(y) for y in zip(*bottom_lip)]
    right_eyebrow = [sum(y) / len(y) for y in zip(*right_eyebrow)]

    # Concatenating average points in the averaged array
    landmark_avg.append(nose_bridge)
    landmark_avg.append(bottom_lip)
    landmark_avg.append(right_eyebrow)

    for box in boxes:
        # extract box corners
        bottom, right, top, left = box
        # initialise the counter for each box
        confirmed_box = 0
        for feature_pos in landmark_avg:
            # extract the feature averaged position
            mean_w, mean_h = feature_pos
            # states if the feature position is inside the face box corners
            if (bottom < mean_h < top
                    and left < mean_w < right):
                # if the feature is inside we increase the counter
                confirmed_box = confirmed_box + 1

        # if every feature is inside the box there is a match
        if confirmed_box == len(landmarks_pos):
            return box

    return None


def find_best_bounding_box(candidate_bounding_boxes, gray_frame):
    # computes the size of the bounding box diagonal
    mean_sizes = (
            np.sum(
                np.array(
                    [
                        [rect.top() - rect.bottom(), rect.left() - rect.right()]
                        for rect in candidate_bounding_boxes
                    ]
                )
                ** 2,
                axis=-1,
            )
            ** 0.5
    )

    # computes the position of the middle of bounding boxes with respect to the middle of the image
    mean_points = np.array(
        [
            [(rect.top() + rect.bottom()) / 2.0, (rect.left() + rect.right()) / 2.0]
            for rect in candidate_bounding_boxes
        ]
    ) - np.array([gray_frame.shape[0] / 2.0, gray_frame.shape[1] / 2.0])

    # computes the distances to center, divided by the bounding box diagonal
    prop_dist = np.sum(mean_points ** 2, axis=-1) ** 0.5 / mean_sizes

    # gets the closer bounding box to the center
    best_bounding_box_id = np.argmin(prop_dist)

    # compute best bounding box
    best_bounding_box = dlib.rectangle(
        int(candidate_bounding_boxes[best_bounding_box_id].left()),
        int(candidate_bounding_boxes[best_bounding_box_id].top()),
        int(candidate_bounding_boxes[best_bounding_box_id].right()),
        int(candidate_bounding_boxes[best_bounding_box_id].bottom()),
    )

    return best_bounding_box


def compress(frame: np.array, compression_factor: float) -> np.array:
    """
    Compress the reference image can affect the efficiency of the matching
    :param compression_factor: if < 1 increase size else reduce size of image
    """
    compressed_shape = (
        int(frame.shape[1] / compression_factor),
        int(frame.shape[0] / compression_factor),
    )
    frame = cv2.resize(frame, compressed_shape)

    return frame


def encodings_read(path):
    data = pickle.loads(open(path, "rb").read())
    return data


def load_ground_truth():
    with open('./data/labelled_videos.json') as json_file:
        data_json = json.load(json_file)
    return data_json


def read_true_name(rec_name, data_json):
    with open('./data/labelled_videos.json') as json_file:
        data_json = json.load(json_file)

    if data_json is None:
        return "Error"
    for real_name in data_json:
        if rec_name in data_json[real_name]:
            return real_name
    return "unknown"
