"""
 Created by Nidhi Mundra on 25/04/17.
"""

import cPickle
import gc
import os

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from feature_generator import FeatureGenerator
from preprocessor import Preprocessor
from scorer import Scorer


class ECGClassifier:
    def __init__(self):
        """
        Initialize all the models and classification pipeline
        """

        # Preprocessor which removes left, right and middle outliers of the wave
        self.preprocessor = Preprocessor()

        # Feature Generator for generating features of a wave
        self.feature_generator = FeatureGenerator()

        # Output arrays
        self.features = []
        self.labels = []
        self.file_names = []

        # Custom scoring module
        self.scorer = make_scorer(Scorer().score, greater_is_better=True)

        # Feature selection model
        self.feature_selector_1 = SelectKBest(f_regression, k=5)
        self.feature_selector_2 = SelectKBest(f_regression, k=5)

        # Classification model
        clf_1 = AdaBoostClassifier()
        base_classifier_1 = RandomForestClassifier()

        params = {
            "base_estimator": [base_classifier_1],
            "n_estimators": range(30, 61, 10),
            "learning_rate": np.arange(0.8, 1.01, 0.05),
        }

        self.classifier_1 = GridSearchCV(clf_1, param_grid=params, cv=10, scoring=self.scorer)

        clf_2 = AdaBoostClassifier()
        base_classifier_2 = RandomForestClassifier()

        params = {
            "base_estimator": [base_classifier_2],
            "n_estimators": range(30, 61, 10),
            "learning_rate": np.arange(0.8, 1.01, 0.05),
        }

        self.classifier_2 = GridSearchCV(clf_2, param_grid=params, cv=10, scoring=self.scorer)

        # Pipeline initialization
        # self.pipeline_1 = Pipeline([('feature_selector', self.feature_selector_1), ('clf', self.classifier_1)])
        # self.pipeline_2 = Pipeline([('feature_selector', self.feature_selector_2), ('clf', self.classifier_2)])

        self.pipeline_1 = self.classifier_1
        self.pipeline_2 = self.classifier_2

    def fit(self, X, Y, filenames):
        """
        Fits the training data and labels in the classifier
        
        :param X: Training data
         
        :param Y: Training labels 
        """

        X1, I1, X2, I2 = self.__transform__(X, filenames, 'training')
        Y1, Y2 = self.__get_feature_labels__(Y, I1, I2)
        self.pipeline_1.fit(X1, Y1)
        self.pipeline_2.fit(X2, Y2)

    def predict(self, X):
        """
        Predict test labels
        
        :param X: Test Data
        
        :return: Return predicted output labels
        """

        X = self.__transform__(X, 'test')
        X1, I1, X2, I2 = self.__transform__(X, 'training')
        Y1 = self.pipeline_1.predict(X1)
        Y2 = self.pipeline_2.predict(X2)
        return self.__merge__(Y1, Y2, I1, I2)

    def score(self, X, Y):
        """
        Predict and compute the accuracy score
        
        :param X: Test data
        
        :param Y: Actual labels of test data 
        """
        predicted_Y = self.predict(X)
        return self.scorer.score(predicted_Y, Y)

    def __transform__(self, X, filenames, prefix='training'):
        """
        Transforms the provided waves data into array containing features of each wave
        
        :param X: 2D array containing data points of all the waves
        
        :return: Tranformed X
        """

        # Return data from pickle files if it was transformed once
        if os.path.isfile("pickle_files/" + prefix + "_peak_data.pickle"):
            # Fetch data points
            with open("pickle_files/" + prefix + "_peak_data.pickle", "rb") as handle:
                peak_features = cPickle.load(handle)

            with open("pickle_files/" + prefix + "_point_data.pickle", "rb") as handle:
                point_features = cPickle.load(handle)

            with open("pickle_files/" + prefix + "_peak_indices.pickle", "rb") as handle:
                peak_indices = cPickle.load(handle)

            with open("pickle_files/" + prefix + "_point_indices.pickle", "rb") as handle:
                point_indices = cPickle.load(handle)

            return [peak_features[1:-1], peak_indices[1:-1], point_features[1:-1], point_indices[1:-1]]

        # Initializing output labels
        peak_features = []
        peak_indices = []
        point_features = []
        point_indices = []

        # for data in X:
        for i in range(0, len(X)):
            data = X[i]
            # try:
            print filenames[i]
            # pyplot.close("all")
            # peakfinder = ([], [])
            # pre.plot("original", data)

            # Remove outlier sections from the wave
            data, outliers = self.preprocessor.process(data)

            # Append the features of the transformed wave in the final output array
            features, type = self.feature_generator.get_features(data, outliers)

            if type == "peak":
                print "peak", i
                peak_features.append(features)
                peak_indices.append(i)
            else:
                print "point", i
                point_features.append(features)
                point_indices.append(i)

        # Store the data in pickle files
        gc.disable()
        with open("pickle_files/" + prefix + '_peak_data.pickle', 'wb') as handle:
            cPickle.dump(peak_features, handle, protocol=cPickle.HIGHEST_PROTOCOL)
        gc.enable()

        gc.disable()
        with open("pickle_files/" + prefix + '_point_data.pickle', 'wb') as handle:
            cPickle.dump(peak_features, handle, protocol=cPickle.HIGHEST_PROTOCOL)
        gc.enable()

        gc.disable()
        with open("pickle_files/" + prefix + '_peak_indices.pickle', 'wb') as handle:
            cPickle.dump(peak_indices, handle, protocol=cPickle.HIGHEST_PROTOCOL)
        gc.enable()

        gc.disable()
        with open("pickle_files/" + prefix + '_point_indices.pickle', 'wb') as handle:
            cPickle.dump(peak_indices, handle, protocol=cPickle.HIGHEST_PROTOCOL)
        gc.enable()

        return [peak_features[1:-1], peak_indices[1:-1], point_features[1:-1], point_indices[1:-1]]

    def __get_feature_labels__(self, Y, I1, I2):
        """
        Get feature labels for corresponding index arrays

        :param Y: Output labels array

        :param I1: Index Array

        :param I2: Index Array

        :return: Corresponding label arrays
        """
        Y1 = []
        Y2 = []

        for i in I1:
            Y1.append(Y[i])

        for i in I2:
            Y2.append(Y[i])

        return [Y1, Y2]

    def __merge__(self, Y1, Y2, I1, I2):

        """
        Merge two output labels arrays using index arrays

        :param Y1: Labels array

        :param Y2: Labels array

        :param I1: Index array

        :param I2: Index array

        :return: Merged output labels array
        """
        output = np.zeros(len(Y1) + len(Y2))

        for i in I1:
            output[i] = Y1[i]

        for i in I2:
            output[i] = Y2[i]

        return output
