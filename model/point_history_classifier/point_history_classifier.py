#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle


class PointHistoryClassifier(object):
    def __init__(
        self,
        model_path='model/point_history_classifier/point_history_classifiergb.pkl',
        score_th=0.5,
        invalid_value=0,
    ):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        if max(self.model.predict_proba([point_history])[0]) < self.score_th:
            return self.invalid_value
        return self.model.predict([point_history])[0]