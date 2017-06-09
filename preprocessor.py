"""
 Created by Jonas Pfeiffer on 26/04/17.
"""

import os.path
import pickle
from operator import itemgetter

import numpy as np
import scipy.io
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


class Preprocessor:
    def __init__(self):
        """
        If a model does not yet exist, that is the is not a model pickle file in "pickle_files"
        the model is trained again and stored as a pickle file in "pickle_files"
        No input, only intitialization of preprocessor models
        """

        self.data = None

        # Labels for flipping, left, right and middle outliers. Only needed if the models do not yet exist
        self.all_labels = None

        # models that are trained for classification of each ECG
        self.flipping_model = self.get_flipping_model()
        self.left_outlier_model = self.get_side_outlier_model(left=0.1, side="left")
        self.right_outlier_model = self.get_side_outlier_model(left=0.9, side="right")
        self.middle_outlier_model = self.get_middle_outlier_model()

    def process(self, data):
        """
        Processes the ECG: Flipps it, deletes obvious outliers at the start, middle and end
        
        :param data:  ECG
        
        :return: preprocessed ECG
        """

        # ECG data
        self.data = data

        # features and prediction of left outliers. left_part contains an array of the indexes that need to be
        # elminated if prediction == 1
        left_feat, left_part = self.get_features_outliers(data, left=0.2)
        try:
            left_pred = self.left_outlier_model.predict(left_feat)
        except:
            left_pred = [1]

        # features and prediction of right outliers. right_part contains an array of the indexes that need to be
        # elminated if prediction == 1
        right_feat, right_part = self.get_features_outliers(data, left=0.9)
        try:
            right_pred = self.right_outlier_model.predict(right_feat)
        except:
            # TODO GOES HERE AT FILE A07331
            right_pred = [1]

        # features and prediction of middle outliers. k_means_points contains an array of the indexes that need to be
        # elminated if prediction == 1
        middle_feat, k_means_data, k_means_points = self.get_features_outliers_middle(data)
        try:
            middle_pred = self.middle_outlier_model.predict(middle_feat)
        except:
            middle_pred = [1]
        # append the the parts that need to be deleted if predicted to contain outliers
        to_delete = []
        if left_pred[0] == 1:
            to_delete.append([0, left_part])
        if right_pred[0] == 1:
            to_delete.append([right_part, len(self.data)])
        if middle_pred[0] == 1:
            # do k-means clustering if the middle part contains outliers to identify which parts need to be eliminated
            test = self.k_means_splitting(k_means_data, k_means_points)
            to_delete += self.k_means_splitting(k_means_data, k_means_points)

        outliers = []
        if to_delete != []:
            outliers = self.delete_parts(to_delete)

        # predict if dataset needs to be flipped on basis of the outlier removed ECG
        flip_feat, base = self.get_features_flipping(data)
        flip_pred = self.flipping_model.predict(flip_feat)

        # if prediction of flipping is true, flip the dataset and return it
        if flip_pred[0] == 1:
            return -self.data, outliers
        else:
            return self.data, outliers

    def get_all_labels(self):
        """
        If all_labels has never been called, load it from the pickle file
        
        :return: all_labels -> the hand labeld set for flipping, left, middle and right outliers
        """
        if self.all_labels == None:
            with open('pickle_files/all_labels.pickle', 'rb') as handle:
                self.all_labels = pickle.load(handle)
        return self.all_labels

    def get_middle_outlier_model(self):

        """
        generates the model to identify if the ECG contains outliers in the middle
        
        :return: middle outlier model
        """

        # If model was already generated once retrieve it from the pickle file
        if os.path.isfile("pickle_files/middle_outlier_model.pickle"):
            with open('pickle_files/middle_outlier_model.pickle', 'rb') as handle:
                return pickle.load(handle)

        # Else train the model
        else:
            print("optimizing middle")
            all_labels = self.get_all_labels()

            features = []
            labels = []
            # Loop through all the labeled files
            for filename, values in list(all_labels.items()):
                mat1 = scipy.io.loadmat('training_data/' + filename)

                # get the ECG
                y = mat1['val'][0]

                # retrieve the features
                feat, k_means_data, k_means_points = self.get_features_outliers_middle(y)

                # append them to one training set
                features.append(feat)

                # retrieve the labeld data and append for training set
                labels.append(int(values["middle"]))

            classifier = AdaBoostClassifier(learning_rate=0.8,
                                            base_estimator=RandomForestClassifier(criterion='gini', n_estimators=5,
                                                                                  max_features=9))
            classifier.fit(features, labels)

            # store the model in pickle file and return the trained model
            with open('pickle_files/middle_outlier_model.pickle', 'wb') as handle:
                pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return classifier

    def get_side_outlier_model(self, left, side):

        if os.path.isfile("pickle_files/" + side + "_outlier_model.pickle"):
            with open("pickle_files/" + side + "_outlier_model.pickle", 'rb') as handle:
                return pickle.load(handle)

        else:
            print(("optimizing side: " + side))
            all_labels = self.get_all_labels()

            features = []
            labels = []
            for filename, values in list(all_labels.items()):
                mat1 = scipy.io.loadmat('training_data/' + filename)
                y = mat1['val'][0]
                feat, left_part = self.get_features_outliers(y, left=left)
                features.append(feat)
                labels.append(int(values[side]))

            if side == "left":
                classifier = AdaBoostClassifier(learning_rate=0.9,
                                                base_estimator=RandomForestClassifier(criterion='gini', n_estimators=5,
                                                                                      max_features=5))
            else:
                classifier = AdaBoostClassifier(learning_rate=1.0,
                                                base_estimator=RandomForestClassifier(criterion='gini', n_estimators=5,
                                                                                      max_features=5))

            classifier.fit(features, labels)

            with open("pickle_files/" + side + "_outlier_model.pickle", 'wb') as handle:
                pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return classifier

    def get_flipping_model(self):

        if os.path.isfile("pickle_files/flipping_model.pickle"):
            with open('pickle_files/flipping_model.pickle', 'rb') as handle:
                return pickle.load(handle)

        else:
            print("optimizing flipping")
            all_labels = self.get_all_labels()

            features = []
            labels = []
            for filename, values in list(all_labels.items()):
                mat1 = scipy.io.loadmat('training_data/' + filename)
                y = mat1['val'][0]
                feat, base = self.get_features_flipping(y)
                features.append(feat)
                labels.append(int(values["flip"]))

            classifier = AdaBoostClassifier(learning_rate=0.79999999999999993,
                                            base_estimator=RandomForestClassifier(criterion='gini', n_estimators=5,
                                                                                  max_features=1))

            classifier.fit(features, labels)

            with open('pickle_files/flipping_model.pickle', 'wb') as handle:
                pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return classifier

    def delete_parts(self, to_delete):
        """
        delets the parts of the ECG that were found to be outliers
        
        :param to_delete: list of arrays start to finish
        
        : return : outliers
        """

        # sort to_delete -> technically should never be unsorted
        to_delete = sorted(to_delete, key=itemgetter(0))

        # overlapping arrays need to be merged: [[0,200],[200,400],[600,800]] = [[0,400],[600,800]]
        new_to_delete = []
        i = 0
        lower = to_delete[i][0]
        upper = to_delete[i][1]
        while True:

            if i == len(to_delete) - 1:
                new_to_delete.append([lower, upper])
                break
            if upper >= to_delete[i + 1][0]:
                upper = to_delete[i + 1][1]
                i += 1
            else:
                new_to_delete.append([lower, upper])
                i += 1
                lower = to_delete[i][0]
                upper = to_delete[i][1]

        # After deleting a part from the ECG data the indices are wrong: update according to distance
        # of last tuple
        distance = 0
        for i in range(0, len(new_to_delete)):
            new_to_delete[i][0] -= distance
            new_to_delete[i][1] -= distance
            distance += new_to_delete[i][1] - new_to_delete[i][0]

        outliers = []
        for elem in new_to_delete:
            outliers.append(elem[0])
            self.data = self.del_snippet(self.data, elem[0], elem[1])
        return outliers


    def del_snippet(self, data, start, end):
        """
        delets a snippet from the ECG data
        
        :param data: ECG
        
        :param start: Starting index that needs to be deleted
        
        :param end: End index that needs to be deleted
        
        :return: updated ECG Data
        """

        length = len(data)
        new_data = []
        if start != 0:
            new_data = np.concatenate((new_data, data[0:start]))
        if end != length:
            new_data = np.concatenate((new_data, data[end:length]))

        return new_data

    def get_kmeans_stats(self, k_means_data, k_means_points, k):

        """
        fit k means cluster with k cluster and get statistics
        
        :param k_means_data: snippets of dataparts
        
        :param k_means_points: range of snippet
        
        :param k: k
        
        :return: statistics to each cluster
        """

        kmeans = KMeans(n_clusters=k)
        prediction = kmeans.fit_predict(k_means_data)
        stats = {}
        for i in range(0, len(prediction)):
            if prediction[i] not in stats:
                stats[prediction[i]] = {}
                stats[prediction[i]]["count"] = 0
                stats[prediction[i]]["points"] = []

            stats[prediction[i]]["count"] += 1
            stats[prediction[i]]["points"].append(k_means_points[i])
        return stats

    def k_means_splitting(self, k_means_data, k_means_points):
        """
        k means clustering und data snippets of middle parts of ECG.
        
        :param k_means_data:  snippets of data parts
        
        :param k_means_points: range of snippet
        
        :return: points to be deleted
        """

        # do the k means clustering
        stats = self.get_kmeans_stats(k_means_data, k_means_points, 2)
        counts = 9999
        delete_points = None

        # evaluate which cluster needs to be added to the to_delete list.
        for key, value in list(stats.items()):

            # if one cluster has less parts included we assume that this is the outlier set, because we are
            # hoping that there are always more non outlier areas than outliers
            if value["count"] < counts:
                counts = value["count"]
                delete_points = value["points"]

            # if we have the case that its even we look at the standard deviation and decide on basis of this.
            elif value["count"] == counts:
                old = []
                new = []
                for array in delete_points:
                    for i in range(array[0], array[1]):
                        old.append(self.data[i])
                for array in value["points"]:
                    for i in range(array[0], array[1]):
                        new.append(self.data[i])
                std_old = np.std(old)
                std_new = np.std(new)
                if std_new > std_old:
                    counts = value["count"]
                    delete_points = value["points"]

        return delete_points

    def get_features_flipping(self, data):
        """
        generate features for flipping
        
        :param data: ECG
        
        :return: Feature array
        """
        maxi = np.max(data)
        mini = np.min(data)
        std = np.std(data)
        sum = np.sum(data)
        up = []
        down = []
        for point in data:
            if point >= 0:
                up.append(point)
            else:
                down.append(point)

        count_up = len(up)
        count_down = len(down)

        if up != []:
            std_up = np.std(up)
        else:
            std_up = 0.0

        if down != []:
            std_down = np.std(down)
        else:
            std_down = 0.0

        sum_up = np.std(up)
        sum_down = np.std(down)

        return [maxi, mini, std, sum, count_up, count_down, std_up, std_down, sum_up, sum_down], int(
            count_down < count_up)

    def get_features_outliers(self, data, left):
        """
        generate features for left and right model
        
        :param data: ECG
        
        :param left: float indicating how the data should be split. left = 0.1 for left side left = 0.9 for right side
        
        :return: Feature array
        """
        length_data = len(data)
        percent_left = left
        left_part = int(round(length_data * percent_left))
        data_left = data[0:left_part]
        data_right = data[left_part:length_data]

        maxi_left = np.max(data_left)
        mini_left = np.min(data_left)
        std_left = np.std(data_left)
        sum_left = np.sum(data_left)
        up_left = []
        down_left = []
        for point_left in data_left:
            if point_left >= 0:
                up_left.append(point_left)
            else:
                down_left.append(point_left)

        count_up_left = len(up_left)
        count_down_left = len(down_left)

        if up_left != []:
            std_up_left = np.std(up_left)
        else:
            std_up_left = 0.0

        if down_left != []:
            std_down_left = np.std(down_left)
        else:
            std_down_left = 0.0

        sum_up_left = np.std(up_left)
        sum_down_left = np.std(down_left)

        maxi_right = np.max(data_right)
        mini_right = np.min(data_right)
        std_right = np.std(data_right)
        sum_right = np.sum(data_right)
        up_right = []
        down_right = []
        for point_right in data_right:
            if point_right >= 0:
                up_right.append(point_right)
            else:
                down_right.append(point_right)

        count_up_right = len(up_right)
        count_down_right = len(down_right)

        if up_right != []:
            std_up_right = np.std(up_right)
        else:
            std_up_right = 0.0

        if down_right != []:
            std_down_right = np.std(down_right)
        else:
            std_down_right = 0.0

        sum_up_right = np.std(up_right)
        sum_down_right = np.std(down_right)

        features = [maxi_left, mini_left, std_left, sum_left, count_up_left, count_down_left, std_up_left,
                    std_down_left, sum_up_left, sum_down_left, maxi_right, mini_right, std_right, sum_right,
                    count_up_right, count_down_right, std_up_right, std_down_right, sum_up_right, sum_down_right]

        return features, left_part

    def get_features_outliers_middle(self, data):
        """
        generate features for middle
        
        :param data: ECG
        
        :return: Feature array
        """
        length_data = len(data)
        percent = 0.1
        part = int(round(length_data * percent))
        start = part
        features = []
        k_means_data = []
        k_means_points = []

        while True:
            points = []
            if start + (1.1 * part) > length_data:
                break
            data_split = data[start:start + part]
            k_means_points.append([start, start + part])
            points.append(np.max(data_split))
            points.append(np.min(data_split))
            points.append(np.std(data_split))
            points.append(np.sum(data_split))

            up = []
            down = []
            for point in data_split:
                if point >= 0:
                    up.append(point)
                else:
                    down.append(point)

            points.append(len(up))
            points.append(len(down))
            if up != []:
                points.append(np.std(up))
            else:
                points.append(0.0)

            if down != []:
                points.append(np.std(down))
            else:
                points.append(0.0)

            if up != []:
                points.append(np.std(up))
            else:
                points.append(0.0)

            if down != []:
                points.append(np.std(down))
            else:
                points.append(0.0)

            start += part
            features += points
            k_means_data.append(points)

        return features, k_means_data, k_means_points
