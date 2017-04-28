import cPickle
import os.path
import pickle
from operator import itemgetter

import numpy as np
import scipy.io
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

class Preprocessor:
    def __init__(self):
        self.data = None
        self.all_labels = None
        self.flipping_model = self.get_flipping_model()
        self.left_outlier_model = self.get_side_outlier_model(left=0.1, side="left")
        self.right_outlier_model = self.get_side_outlier_model(left=0.9, side="right")
        self.middle_outlier_model = self.get_middle_outlier_model()

    def process(self, data):

        self.data = data

        left_feat, left_part = self.get_features_outliers(data, left=0.2)
        left_pred = self.left_outlier_model.predict(left_feat)

        right_feat, right_part = self.get_features_outliers(data, left=0.9)
        right_pred = self.right_outlier_model.predict(right_feat)

        middle_feat, k_means_data, k_means_points = self.get_features_outliers_middle(data)
        middle_pred = self.middle_outlier_model.predict(middle_feat)

        to_delete = []

        if left_pred[0] == 1:
            to_delete.append([0, left_part])
        if right_pred[0] == 1:
            to_delete.append([right_part, len(self.data)])
        if middle_pred[0] == 1:
            test = self.k_means_splitting(k_means_data, k_means_points)
            to_delete += self.k_means_splitting(k_means_data, k_means_points)
        if (to_delete != []):
            self.delete_parts(to_delete)

        flip_feat, base = self.get_features_flipping(data)
        flip_pred = self.flipping_model.predict(flip_feat)

        if (flip_pred[0] == 1):
            print("here")
            return -self.data
        else:
            return self.data

    def get_all_labels(self):
        if self.all_labels == None:
            with open('pickle_files/all_labels.pickle', 'rb') as handle:
                self.all_labels = pickle.load(handle)
        return self.all_labels

    def get_middle_outlier_model(self):

        if os.path.isfile("pickle_files/middle_outlier_model.pickle"):
            with open('pickle_files/middle_outlier_model.pickle', 'rb') as handle:
                return pickle.load(handle)

        else:
            print "optimizing middle"
            all_labels = self.get_all_labels()

            features = []
            labels = []
            baseline_flip = []
            for filename, values in all_labels.iteritems():
                mat1 = scipy.io.loadmat('training_data/' + filename)
                y = mat1['val'][0]
                # feat, base = self.get_features_flipping(y)
                # feat = get_features_outliers(y, left= 0.9)
                feat, k_means_data, k_means_points = self.get_features_outliers_middle(y)
                features.append(feat)
                labels.append(int(values["middle"]))

            # with open('middle_outliers_features.pickle', 'rb') as handle:
            #     features = pickle.load(handle)
            #
            # with open('middle_outliers_labels.pickle', 'rb') as handle:
            #     labels = pickle.load(handle)


            # classifier = LogisticRegression()
            #
            # base_classifier = RandomForestClassifier()

            classifier = AdaBoostClassifier(
             n_estimators= 60,
             base_estimator= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                                                      min_impurity_split=1e-07, min_samples_leaf=1,
                                                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                      n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
                                                      verbose=0, warm_start=False),
             learning_rate= 0.95000000000000018)
            # 0.869607843137
            # classifier = AdaBoostClassifier(base_estimator=RandomForestClassifier())
            # params = {
            #     # "penalty": ["l1","l2"],
            #     # "dual": [True,False],
            #     #  "solver": [ "lbfgs", "liblinear", "sag"]
            #
            #
            #
            #     # Adaboost
            #     # "base_estimator": [base_classifier],
            #     "n_estimators": range(30, 61, 10),
            #     "learning_rate": np.arange(0.8, 1.01, 0.05),
            #     "base_estimator__n_estimators": range(5, 15, 5),
            #     "base_estimator__criterion": ["gini", "entropy"],
            #     "base_estimator__max_features": range(1, 13, 4)
            #
            # }

            # new_features = np.array(new_features)
            # new_labels = np.array(new_labels)

            # cv = GridSearchCV(classifier, param_grid=params, cv=10)
            # cv.fit(features, labels)
            # print "middle optimization"
            # print cv.best_params_
            # print cv.best_score_
            # classifier = cv.best_estimator_
            classifier.fit(features, labels)

            with open('pickle_files/middle_outlier_model.pickle', 'wb') as handle:
                cPickle.dump(classifier, handle, protocol=cPickle.HIGHEST_PROTOCOL)
            return classifier

    def get_side_outlier_model(self, left, side):

        if os.path.isfile("pickle_files/" + side + "_outlier_model.pickle"):
            with open("pickle_files/" + side + "_outlier_model.pickle", 'rb') as handle:
                return pickle.load(handle)

        else:
            print "optimizing side" + side
            all_labels = self.get_all_labels()

            features = []
            labels = []
            baseline_flip = []
            for filename, values in all_labels.iteritems():
                mat1 = scipy.io.loadmat('training_data/' + filename)
                y = mat1['val'][0]
                # feat, base = self.get_features_flipping(y)
                feat, left_part = self.get_features_outliers(y, left=left)
                # feat, k_means_data, k_means_points = get_features_outliers_middle(y)
                features.append(feat)
                labels.append(int(values[side]))

            #
            # with open('left_outliers_features.pickle', 'rb') as handle:
            #     features = pickle.load(handle)
            #
            # with open('left_outliers_labels.pickle', 'rb') as handle:
            #     labels = pickle.load(handle)

            classifier = AdaBoostClassifier(n_estimators= 60,
                                                      base_estimator =  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                                                      min_impurity_split=1e-07, min_samples_leaf=1,
                                                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                      n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
                                                      verbose=0, warm_start=False),
                                                      learning_rate=  0.85000000000000009)

            classifier.fit(features,labels)

            # classifier = AdaBoostClassifier(base_estimator=RandomForestClassifier())
            # params = {
            #     # "penalty": ["l1","l2"],
            #     # "dual": [True,False],
            #     #  "solver": [ "lbfgs", "liblinear", "sag"]
            #
            #
            #
            #     # Adaboost
            #     # "base_estimator": [base_classifier],
            #     "n_estimators": range(30, 61, 10),
            #     "learning_rate": np.arange(0.8, 1.01, 0.05),
            #     "base_estimator__n_estimators": range(5, 15, 5),
            #     "base_estimator__criterion": ["gini", "entropy"],
            #     "base_estimator__max_features": range(1, 13, 4)
            #
            # }
            #
            # # new_features = np.array(new_features)
            # # new_labels = np.array(new_labels)
            #
            # cv = GridSearchCV(classifier, param_grid=params, cv=10)
            # cv.fit(features, labels)
            # print "side optimization"
            # print cv.best_params_
            # print cv.best_score_
            # classifier = cv.best_estimator_
            # classifier.fit(features, labels)

            with open("pickle_files/" + side + "_outlier_model.pickle", 'wb') as handle:
                cPickle.dump(classifier, handle, protocol=cPickle.HIGHEST_PROTOCOL)
            return classifier

    def get_flipping_model(self):

        if os.path.isfile("pickle_files/flipping_model.pickle"):
            with open('pickle_files/flipping_model.pickle', 'rb') as handle:
                return pickle.load(handle)

        else:
            print "optimizing flipping"
            all_labels = self.get_all_labels()

            features = []
            labels = []
            baseline_flip = []
            for filename, values in all_labels.iteritems():
                mat1 = scipy.io.loadmat('training_data/' + filename)
                y = mat1['val'][0]
                feat, base = self.get_features_flipping(y)
                # feat = get_features_outliers(y, left= 0.9)
                # feat, k_means_data, k_means_points = get_features_outliers_middle(y)
                features.append(feat)
                labels.append(int(values["flip"]))

            classifier = LogisticRegression()
            classifier.fit(features, labels)

            # classifier = AdaBoostClassifier(base_estimator=RandomForestClassifier())
            # params = {
            #     # "penalty": ["l1","l2"],
            #     # "dual": [True,False],
            #     #  "solver": [ "lbfgs", "liblinear", "sag"]
            #
            #
            #
            #     # Adaboost
            #     # "base_estimator": [base_classifier],
            #     "n_estimators": range(30, 61, 10),
            #     "learning_rate": np.arange(0.8, 1.01, 0.05),
            #     "base_estimator__n_estimators": range(5, 15, 5),
            #     "base_estimator__criterion": ["gini", "entropy"],
            #     "base_estimator__max_features": range(1, 10, 2)
            #
            # }
            #
            # # new_features = np.array(new_features)
            # # new_labels = np.array(new_labels)
            #
            # cv = GridSearchCV(classifier, param_grid=params, cv=10)
            # cv.fit(features, labels)
            # print "flip optimization"
            # print cv.best_params_
            # print cv.best_score_
            # classifier = cv.best_estimator_
            # classifier.fit(features, labels)

            with open('pickle_files/flipping_model.pickle', 'wb') as handle:
                cPickle.dump(classifier, handle, protocol=cPickle.HIGHEST_PROTOCOL)
            return classifier

    def delete_parts(self, to_delete):
        to_delete = sorted(to_delete, key=itemgetter(0))
        new_to_delete = []
        i = 0
        j = 1
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
        distance = 0
        for i in range(0, len(new_to_delete)):
            new_to_delete[i][0] -= distance
            new_to_delete[i][1] -= distance
            distance += new_to_delete[i][1] - new_to_delete[i][0]

        outliers = []
        for elem in new_to_delete:
            outliers.append(elem[0])
            self.data = self.del_snippet(self.data, elem[0], elem[1])

            # return new_to_delete

    def del_snippet(self, data, start, end):
        length = len(data)
        new_data = []
        if (start != 0):
            # test = data[0:start]
            new_data = np.concatenate((new_data, data[0:start]))
        if (end != length):
            new_data = np.concatenate((new_data, data[end:length]))

        return new_data

    def get_kmeans_stats(self, k_means_data, k_means_points, k):
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
        stats = self.get_kmeans_stats(k_means_data, k_means_points, 2)
        counts = 9999
        delete_points = None
        for key, value in stats.iteritems():
            if value["count"] < counts:
                counts = value["count"]
                delete_points = value["points"]
        return delete_points

    def get_features_flipping(self, data):
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
        std_up = np.std(up)
        std_down = np.std(down)
        sum_up = np.std(up)
        sum_down = np.std(down)

        return [maxi, mini, std, sum, count_up, count_down, std_up, std_down, sum_up, sum_down], int(
            count_down < count_up)

    def get_features_outliers(self, data, left):
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
        std_up_left = np.std(up_left)
        std_down_left = np.std(down_left)
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
        std_up_right = np.std(up_right)
        std_down_right = np.std(down_right)
        sum_up_right = np.std(up_right)
        sum_down_right = np.std(down_right)

        features = [maxi_left, mini_left, std_left, sum_left, count_up_left, count_down_left, std_up_left,
                    std_down_left, sum_up_left, sum_down_left, maxi_right, mini_right, std_right, sum_right,
                    count_up_right, count_down_right, std_up_right, std_down_right, sum_up_right, sum_down_right]

        return features, left_part

    def get_features_outliers_middle(self, data):
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
            points.append(np.std(up))
            points.append(np.std(down))
            points.append(np.std(up))
            points.append(np.std(down))
            start += part
            features += points
            k_means_data.append(points)

        return features, k_means_data, k_means_points

#
#
# def delete_parts( to_delete):
#     to_delete = sorted(to_delete, key=itemgetter(0))
#     new_to_delete = []
#     i = 0
#     j = 1
#     lower = to_delete[i][0]
#     upper = to_delete[i][1]
#     while True:
#
#         if i == len(to_delete) - 1:
#             new_to_delete.append([lower, upper])
#             break
#         if upper >= to_delete[i + 1][0]:
#             upper = to_delete[i + 1][1]
#             i += 1
#         else:
#             new_to_delete.append([lower, upper])
#             i += 1
#             lower = to_delete[i][0]
#             upper = to_delete[i][1]
#     distance = 0
#     for i in range(0, len(new_to_delete)):
#         new_to_delete[i][0] -= distance
#         new_to_delete[i][1] -= distance
#         distance += new_to_delete[i][1] - new_to_delete[i][0]
#
#     return new_to_delete
#
#
# to_delete = [ [0,200], [200, 400], [600, 700], [699, 800], [1000,1002] ]
# new_to_delete = delete_parts(to_delete)
#

#
# preprocessing = Preprocessor()
#
#
#
#
# with open('all_labels.pickle', 'rb') as handle:
#     all_labels = pickle.load(handle)
#
# features = []
# labels = []
# baseline_flip = []
# for filename, values in all_labels.iteritems():
#
#
#     mat1 = scipy.io.loadmat('training_data/' + filename)
#     y = mat1['val'][0]
#     # feat, base = get_features_flipping(y)
#     # feat = get_features_outliers(y, left= 0.9)
#     feat, k_means_data, k_means_points = get_features_outliers_middle(y)
#     features.append(feat)
#     labels.append(int(values["middle"]))
#
# # with open('left_outliers_features.pickle', 'wb') as handle:
# #     cPickle.dump(features, handle, protocol=cPickle.HIGHEST_PROTOCOL)
# #
# # with open('left_outliers_labels.pickle', 'wb') as handle:
# #     cPickle.dump(labels, handle, protocol=cPickle.HIGHEST_PROTOCOL)
# #
# # with open('right_outliers_features.pickle', 'wb') as handle:
# #     cPickle.dump(features, handle, protocol=cPickle.HIGHEST_PROTOCOL)
# #
# # with open('right_outliers_labels.pickle', 'wb') as handle:
# #     cPickle.dump(labels, handle, protocol=cPickle.HIGHEST_PROTOCOL)
#
# with open('middle_outliers_features.pickle', 'wb') as handle:
#     cPickle.dump(features, handle, protocol=cPickle.HIGHEST_PROTOCOL)
#
# with open('middle_outliers_labels.pickle', 'wb') as handle:
#     cPickle.dump(labels, handle, protocol=cPickle.HIGHEST_PROTOCOL)
#
#
# # with open('flip_features.pickle', 'wb') as handle:
# #     cPickle.dump(features, handle, protocol=cPickle.HIGHEST_PROTOCOL)
# #
# # with open('flip_labels.pickle', 'wb') as handle:
# #     cPickle.dump(labels, handle, protocol=cPickle.HIGHEST_PROTOCOL)
# #
# # with open('flip_labels_baseline.pickle', 'wb') as handle:
# #     cPickle.dump(baseline_flip, handle, protocol=cPickle.HIGHEST_PROTOCOL)
#
#
#
# # with open('flip_features.pickle', 'rb') as handle:
# #     features = pickle.load(handle)
# #
# #
# # with open('flip_labels.pickle', 'rb') as handle:
# #     labels = pickle.load(handle)
# #
# #
# # with open('flip_labels_baseline.pickle', 'rb') as handle:
# #     baseline_flip = pickle.load(handle)
#
#
#
#
# with open('right_outliers_features.pickle', 'rb') as handle:
#     features = pickle.load(handle)
#
#
# with open('right_outliers_labels.pickle', 'rb') as handle:
#     labels = pickle.load(handle)
#
#
#
#
# # print "baseline = " + str(accuracy_score(labels, baseline_flip))
#
# # features = normalize(features, norm = "max" )
# #
#
#
# classifier = AdaBoostClassifier()
# # classifier = LogisticRegression()
#
# base_classifier = RandomForestClassifier()
#
# params = {
#     # "penalty": ["l1","l2"],
#     # "dual": [True,False],
#     #  "solver": [ "lbfgs", "liblinear", "sag"]
#
#
#
#     # Adaboost
#     "base_estimator": [base_classifier],
#     "n_estimators": range(30, 61, 10),
#     # "learning_rate": np.arange(0.8, 1.01, 0.05),
#     # "base_estimator__n_estimators": range(5,15,5),
#     # "base_estimator__criterion": ["gini", "entropy"],
#     # "base_estimator__max_features":
#
# }
#
# # new_features = np.array(new_features)
# # new_labels = np.array(new_labels)
#
# cv = GridSearchCV(classifier, param_grid=params, cv=10)
# cv.fit(features, labels)
# print cv.best_params_
# print cv.best_score_





#
# {'n_estimators': 40, 'base_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
#             verbose=0, warm_start=False)}
# 0.961764705882
