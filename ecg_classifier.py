"""
 Created by Nidhi Mundra on 25/04/17.
"""

import gc
import os
import pickle

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

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
        self.scorer = Scorer()

        # Feature selection models
        self.feature_selector_1 = SelectFromModel(LinearSVC(penalty="l1", dual=False))
        self.feature_selector_2 = SelectFromModel(LinearSVC(penalty="l1", dual=False))

        # Classification models
        # clf_1 = DecisionTreeClassifier()
        #
        # params = {"criterion": ["gini", "entropy"],
        #           "min_samples_split": [2, 10, 20],
        #           "max_depth": [None, 2, 5, 10],
        #           "min_samples_leaf": [1, 5, 10],
        #           "max_leaf_nodes": [None, 5, 10, 20],
        #           }

        clf_1 = AdaBoostClassifier()
        base_classifier_1 = RandomForestClassifier()

        # Best Classifier 1
        # {'n_estimators': 40,
        #  'base_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        #                                           max_depth=None, max_features='auto', max_leaf_nodes=None,
        #                                           min_impurity_split=1e-07, min_samples_leaf=1,
        #                                           min_samples_split=2, min_weight_fraction_leaf=0.0,
        #                                           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
        #                                           verbose=0, warm_start=False), 'learning_rate': 0.85000000000000009}
        # Best
        # Classifier
        # 2
        # {'n_estimators': 30,
        #  'base_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        #                                           max_depth=None, max_features='auto', max_leaf_nodes=None,
        #                                           min_impurity_split=1e-07, min_samples_leaf=1,
        #                                           min_samples_split=2, min_weight_fraction_leaf=0.0,
        #                                           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
        #                                           verbose=0, warm_start=False), 'learning_rate': 0.80000000000000004}

        # params = {
        #     "base_estimator": [base_classifier_1],
        #     "n_estimators": range(30, 61, 10),
        #     "learning_rate": np.arange(0.8, 1.01, 0.05),
        # }
        optimal_params = {
            "base_estimator": [base_classifier_1],
            'n_estimators': [40],
            'learning_rate': [0.85000000000000009]
        }

        self.classifier_1 = GridSearchCV(clf_1, param_grid=optimal_params,
                                         cv=10, scoring=make_scorer(self.scorer.score), verbose=10)

        # clf_2 = DecisionTreeClassifier()
        clf_2 = AdaBoostClassifier()
        base_classifier_2 = RandomForestClassifier()

        # params = {
        #     "base_estimator": [base_classifier_2],
        #     "n_estimators": range(30, 61, 10),
        #     "learning_rate": np.arange(0.8, 1.01, 0.05),
        # }

        optimal_params = {
            "base_estimator": [base_classifier_2],
            'n_estimators': [30],
            'learning_rate': [0.80000000000000004]
        }

        # params = {"criterion": ["gini", "entropy"],
        #           "min_samples_split": [2, 10, 20],
        #           "max_depth": [None, 2, 5, 10],
        #           "min_samples_leaf": [1, 5, 10],
        #           "max_leaf_nodes": [None, 5, 10, 20],
        #           }
        self.classifier_2 = GridSearchCV(clf_2, param_grid=optimal_params,
                                         cv=2, scoring=make_scorer(self.scorer.score), verbose=10)

        # Pipeline initializations
        self.pipeline_1 = Pipeline([('feature_selector', self.feature_selector_1), ('clf', self.classifier_1)])
        self.pipeline_2 = Pipeline([('feature_selector', self.feature_selector_2), ('clf', self.classifier_2)])

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

    def predict(self, X, file_names):
        """
        Predict test labels
        
        :param X: Test Data
        
        :return: Return predicted output labels
        """

        X1, I1, X2, I2 = self.__transform__(X, file_names, 'test')
        Y1 = self.pipeline_1.predict(X1)
        Y2 = self.pipeline_2.predict(X2)
        return self.__merge__(Y1, Y2, I1, I2)

    def score(self, X, Y, file_names):
        """
        Predict and compute the accuracy score
        
        :param X: Test data
        
        :param Y: Actual labels of test data 
        """
        predicted_Y = self.predict(X, file_names)
        return self.scorer.score(predicted_Y, Y)

    def __replace_missing_values__(self, matrix, value):
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                # if matrix[i][j] == None:
                #     matrix[i][j] = np.mean(matrix[:i])
                # matrix[i][j] = float(matrix[i][j])
                if np.isnan(matrix[i][j]):
                    matrix[i][j] = np.nanmean(matrix[:j])
                if not np.isfinite(matrix[i][j]):
                    matrix[i][j] = np.nanmean(matrix[:j])
        return matrix




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
                peak_features = pickle.load(handle)

            with open("pickle_files/" + prefix + "_point_data.pickle", "rb") as handle:
                point_features = pickle.load(handle)

            with open("pickle_files/" + prefix + "_peak_indices.pickle", "rb") as handle:
                peak_indices = pickle.load(handle)

            with open("pickle_files/" + prefix + "_point_indices.pickle", "rb") as handle:
                point_indices = pickle.load(handle)
            peak_features = self.__replace_missing_values__(peak_features, "average")
            return [peak_features, peak_indices, point_features, point_indices]

        # Initializing output labels
        peak_features = []
        peak_indices = []
        point_features = []
        point_indices = []

        # for data in X:
        for i in range(0, len(X)):
            data = X[i]

            print((filenames[i]))
            # pyplot.close("all")
            # peakfinder = ([], [])
            # pre.plot("original", data)

            # Remove outlier sections from the wave
            data, outliers = self.preprocessor.process(data)

            # Append the features of the transformed wave in the final output array
            features, type = self.feature_generator.get_features(data, outliers)

            if type == "peak":
                peak_features.append(features)
                peak_indices.append(i)
            else:
                point_features.append(features)
                point_indices.append(i)

        # Store the data in pickle files
        gc.disable()
        with open("pickle_files/" + prefix + '_peak_data.pickle', 'wb') as handle:
            pickle.dump(peak_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        gc.enable()

        gc.disable()
        with open("pickle_files/" + prefix + '_point_data.pickle', 'wb') as handle:
            pickle.dump(point_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        gc.enable()

        gc.disable()
        with open("pickle_files/" + prefix + '_peak_indices.pickle', 'wb') as handle:
            pickle.dump(peak_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        gc.enable()

        gc.disable()
        with open("pickle_files/" + prefix + '_point_indices.pickle', 'wb') as handle:
            pickle.dump(point_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        gc.enable()

        return [peak_features, peak_indices, point_features, point_indices]

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

        for i in range(0, len(I1)):
            output[I1[i]] = Y1[i]

        for i in range(0, len(I2)):
            output[I2[i]] = Y2[i]

        return output
