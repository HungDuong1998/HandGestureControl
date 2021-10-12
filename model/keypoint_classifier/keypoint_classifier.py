#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifiergb.pkl'
    ):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def __call__(
        self,
        landmark_list,
    ):
        return self.model.predict([landmark_list])[0]
