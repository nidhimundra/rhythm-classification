import cPickle
import gc
import os

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
        self.feature_selector = SelectKBest(f_regression, k=5)

        # Classification model
        clf = AdaBoostClassifier()
        base_classifier = RandomForestClassifier()

        params = {
            "base_estimator": [base_classifier],
            "n_estimators": range(30, 61, 10),
            "learning_rate": np.arange(0.8, 1.01, 0.05),
        }

        self.classifier = GridSearchCV(clf, param_grid=params, cv=10, scoring=self.scorer)

        # Pipeline initialization
        self.pipeline = Pipeline([('feature_selector', self.feature_selector), ('clf', self.classifier)])

    def fit(self, X, Y):
        """
        Fits the training data and labels in the classifier
        
        :param X: Training data
         
        :param Y: Training labels 
        """

        X = self.__transform__(X, 'training')
        self.pipeline.fit(X, Y)

    def predict(self, X):
        """
        Predict test labels
        
        :param X: Test Data
        
        :return: Return predicted output labels
        """

        X = self.__transform__(X, 'test')
        return self.pipeline.predict(X)

    def score(self, X, Y):
        """
        Predict and compute the accuracy score
        
        :param X: Test data
        
        :param Y: Actual labels of test data 
        """
        predicted_Y = self.predict(X)
        return self.scorer.score(predicted_Y, Y)

    def __transform__(self, X, prefix='training'):
        """
        Transforms the provided waves data into array containing features of each wave
        
        :param X: 2D array containing data points of all the waves
        
        :return: Tranformed X
        """

        # Return data from pickle files if it was transformed once
        if os.path.isfile("pickle_files/" + prefix + "_data.pickle"):
            # Fetch data points
            with open("pickle_files/" + prefix + "_data.pickle", "rb") as handle:
                transformed_X = cPickle.load(handle)

            return transformed_X

        # Initializing output labels
        transformed_X = []

        for data in X:
            # try:

            # Remove outlier sections from the wave
            data, outliers = self.preprocessor.process(data)

            # Append the features of the transformed wave in the final output array
            transformed_X.append(self.feature_generator.get_features(data, outliers))
            # except:
            #     pyplot.close("all")
            #     peakfinder = PeakFinder(data, [])
            #     peakfinder.plot("original")
            #     # Append zeros in case of erroneous wave
            #     transformed_X.append(np.zeros(400))
                # TODO: Do normal classification here - use rolling mean, std, var, max, min, etc  - Nidhi

        # Store the data in pickle files
        gc.disable()
        with open("pickle_files/" + prefix + '_data.pickle', 'wb') as handle:
            cPickle.dump(transformed_X, handle, protocol=cPickle.HIGHEST_PROTOCOL)
        gc.enable()

        return transformed_X
