#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import pyautogui
import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier
from model import PointHistoryClassifier

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    wCam = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    hCam = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    wScr, hScr = pyautogui.size()

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    point_history_classifier_fullhand = PointHistoryClassifier(model_path='model/point_history_classifier/point_history_fullhand_classifierlr.pkl')

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label_fullhand.csv',
            encoding='utf-8-sig') as f:
        point_history_fullhand_classifier_labels = csv.reader(f)
        point_history_fullhand_classifier_labels = [
            row[0] for row in point_history_fullhand_classifier_labels
        ]

    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    finger_gesture_history_fullhand = deque(maxlen=history_length)
    
    plocX,clocX,plocY,clocY = None,None,None,None
    plocwX,clocwX,plocwY,clocwY = None,None,None,None
    ptic = pticswipe = cv.getTickCount()
    ticfreq = cv.getTickFrequency()

    valid_length = 4
    gesturevalid = deque(maxlen=valid_length)

    while True:

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            continue
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 1:  # Point gesture
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history,mode=1)
                    point_history.append(landmark_list[8]+landmark_list[0])
                else:
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history,mode=2)
                    point_history.append(landmark_list[12] + landmark_list[9]  +  # middle
                                         landmark_list[0] # wrist 
                                        )

                # Finger gesture classification
                finger_gesture_id = 0
                finger_gesture_id_fullhand = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 4):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                if point_history_len == (history_length * 6):
                    finger_gesture_id_fullhand = point_history_classifier_fullhand(pre_processed_point_history_list)
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()
                finger_gesture_history_fullhand.append(finger_gesture_id_fullhand)
                most_common_fg_id_fullhand = Counter(finger_gesture_history_fullhand).most_common()         
                if hand_sign_id == 1:  # Point gesture
                    if point_history_classifier_labels[most_common_fg_id[0][0]] == "Move" and point_history_classifier_labels[finger_gesture_id] == "Move":
                        clocX, clocY = landmark_list[8]
                        clocwX, clocwY = landmark_list[0]
                        if plocX == None:
                            plocX = clocX
                        if plocY == None:
                            plocY = clocY
                        if plocwX == None:
                            plocwX = clocwX
                        if plocwY == None:
                            plocwY = clocwY
                        distX,distY = pathplanning(clocX-plocX,clocY-plocY,wCam,hCam,wScr,hScr,noise=0,pathingalgorithm=True,rescale=True)
                        _,distwY = pathplanning(clocwX-plocwX,clocwY-plocwY,wCam,hCam,wScr,hScr,noise=0,pathingalgorithm=True,rescale=True)
                        movement, movementinv = 1,1
                        if distwY != 0: 
                            movement = abs(distY/distwY)
                        if distY != 0: 
                            movementinv = abs(distwY/distY)
                        if movement <= 2 and movementinv >= 0.5:
                            mouseX, mouseY =  pyautogui.position()
                            pyautogui.moveTo(mouseX + distX, mouseY + distY)
                        plocX, plocY = clocX, clocY
                        plocwX, plocwY = clocwX, clocwY
                    elif point_history_classifier_labels[most_common_fg_id[0][0]] == "Click":
                        plocX, plocY = None,None        
                        ctic = cv.getTickCount()
                        timeinsecond = (ctic-ptic)/ticfreq
                        if timeinsecond > 0.5:
                            pyautogui.mouseDown()
                            pyautogui.mouseUp()
                        ptic = ctic
                    else:
                        plocX, plocY = None,None
                else:
                    gesturevalid.append(point_history_fullhand_classifier_labels[most_common_fg_id_fullhand[0][0]])
                    if gesturevalid.count("Swipe Right") == gesturevalid.maxlen:
                        cticswipe = cv.getTickCount()
                        timeinsecond = (cticswipe-pticswipe)/ticfreq
                        if timeinsecond > 0.5:
                            pyautogui.press("right")      
                        pticswipe = cticswipe
                        gesturevalid.clear()
                    if gesturevalid.count("Swipe Left") == gesturevalid.maxlen:
                        cticswipe = cv.getTickCount()
                        timeinsecond = (cticswipe-pticswipe)/ticfreq
                        if timeinsecond > 0.5:
                            pyautogui.press("left")      
                        pticswipe = cticswipe
                        gesturevalid.clear()
        else:
            point_history.append([0])


def pathplanning(distX,distY,wCam,hCam,wScr,hScr,noise = 2, pathingalgorithm=False,rescale=False):
    # 1. Noise supression
    if -noise < distX < noise: distX = distX/10
    if -noise < distY < noise: distY = distY/10
    # 2. Rescale
    if rescale:
        distX = distX*wScr/wCam
        distY = distY*hScr/hCam
    # 3. Path planning
    if pathingalgorithm:
        if abs(distX) > 30: distX = 2*distX
        if abs(distY) > 30: distY = 2*distY
    return distX, distY


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y  = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history,mode=1):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    if mode == 1:
        # Convert to relative coordinates
        base_x, base_y, base_x1, base_y1 = 0, 0, 0, 0
        for index, point in enumerate(temp_point_history):
            if len(point) != 4: 
                point = np.zeros(4)
                temp_point_history[index] = np.zeros(4)
            if index == 0:
                base_x, base_y , base_x1, base_y1 = point[0], point[1], point[2], point[3]

            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
            temp_point_history[index][2] = (temp_point_history[index][2] - base_x1) / image_width
            temp_point_history[index][3] = (temp_point_history[index][3] - base_y1) / image_height

    if mode == 2:
        # Convert to relative coordinates
        base_x, base_y, base_x1, base_y1 = 0, 0, 0, 0
        base_x2, base_y2 = 0, 0
        for index, point in enumerate(temp_point_history):
            if len(point) != 6: 
                point = np.zeros(6)
                temp_point_history[index] = np.zeros(6)
            if index == 0:
                base_x, base_y , base_x1, base_y1 = point[0], point[1], point[2], point[3]
                base_x2, base_y2 = point[4], point[5]
            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
            temp_point_history[index][2] = (temp_point_history[index][2] - base_x1) / image_width
            temp_point_history[index][3] = (temp_point_history[index][3] - base_y1) / image_height
            temp_point_history[index][4] = (temp_point_history[index][4] - base_x2) / image_width
            temp_point_history[index][5] = (temp_point_history[index][5] - base_y2) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


if __name__ == '__main__':
    main()
