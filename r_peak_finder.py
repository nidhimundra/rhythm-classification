import copy

import numpy as np
from matplotlib import pyplot
from sklearn.cluster import KMeans

import peakutils
from peakutils.plot import plot as pplot


class RPeakFinder:
    def __init__(self, data):
        pyplot.close("all")
        # self.data = data
        self.data = self.__flip_data__(data)
        self.peaks = self.__initial_peaks__()
        self.outliers = []
        self.r_peaks = copy.copy(self.peaks)

    def __flip_data__(self, data):
        # minus = 0
        # plus = 0
        # for i in data:
        #     if i > 200:
        #         plus += 1
        #     if i < -200:
        #         minus += 1
        # if minus > plus:
        #     return -data
        # else:
        #     return data
        # data_sum = 0
        # for point in data:
        #     if point > -200 and point < 200:
        #         data_sum += point



        data_sum = np.sum(data)
        if data_sum >= 0:
            return data
        else:
            return -data




    def __initial_peaks__(self):
        return peakutils.indexes(self.data, thres=0.46, min_dist=30)

    def r_detection_outlier_removal(self):

        self.find_r_peaks(outliers_removal=True)
        self.remove_outliers(self.outliers)
        self.find_r_peaks(outliers_removal=False)

    def get_kmeans_stats(self, data, x, k):
        kmeans = KMeans(n_clusters=k)
        prediction = kmeans.fit_predict(x)
        stats = {}
        for i in range(0, len(data)):
            if prediction[i] not in stats:
                stats[prediction[i]] = {}
                stats[prediction[i]]["values"] = []
                stats[prediction[i]]["count"] = 0
            stats[prediction[i]]["count"] += 1
            stats[prediction[i]]["values"].append(data[i])
        for key, value in stats.iteritems():
            stats[key]["std"] = np.std(value["values"])
            stats[key]["mean"] = kmeans.cluster_centers_[key]

        return stats

    def find_r_peaks(self, outliers_removal=False):

        # data = copy.deepcopy(self.data)

        # peaks = copy.deepcopy(self.r_peaks)
        # self.r_peaks = self.__initial_peaks__()

        all_peaks_values = []
        for i in self.r_peaks:
            all_peaks_values.append(self.data[i])

        x = np.array([all_peaks_values]).T
        std_all = np.std(all_peaks_values)
        median_all = np.median(all_peaks_values)
        try:
            # k=1
            r_k = None
            r_k_center = 0
            std_k = 99
            for k in range(1, 4):
                # kmeans = KMeans(n_clusters=k)
                # prediction = kmeans.fit_predict(x)
                stats = self.get_kmeans_stats(all_peaks_values, x, k)

                for key, value in stats.iteritems():
                    # stats[key]["std"] = np.std(value["values"])
                    # stats[key]["mean"] = kmeans.cluster_centers_[key]
                    std_perc = stats[key]["std"] / stats[key]["mean"]
                    a_first_left = float(stats[key]["count"]) / (len(all_peaks_values))
                    a_first_right = 1.0 / (2 * k)
                    a_first = (float(stats[key]["count"]) / (len(all_peaks_values)) > (1.0 / (2 * k)))
                    a_second = stats[key]["mean"] > r_k_center
                    a_third = stats[key]["std"] / stats[key]["mean"] < 1.2 * std_k

                    # print"Her"
                    if (float(stats[key]["count"]) / (len(all_peaks_values)) > (1.0 / (2 * k))) and \
                                    stats[key]["mean"] > r_k_center and stats[key]["std"] / stats[key][
                        "mean"] < 1.2 * std_k:
                        r_k = key
                        median_all = stats[key]["mean"]
                        std_k = stats[key]["std"] / stats[key]["mean"]

            new_r_peaks = []

            # for i in range(0,len(prediction)):
            #     if prediction[i] == r_k:
            #         new_r_peaks.append(all_peaks_values[i])
            # self.r_peaks = new_r_peaks





        except:
            print"to less"


            # stats[prediction[i]][1] += 1
            # if stats[prediction[i]][2] == 0:
            #     stats[prediction[i]][2] = []
            # stats[prediction[i]][2].append(all_peaks_values[i])

        # for i in range(0,k):
        #     stats[i][3] = np.std(stats[i][2])
        #     stats[i][4] = kmeans.cluster_centers_[i]

        #
        outliers = [0]
        # r_peaks = []
        other_peaks = []

        # find NOT outliers but not R Peaks:
        previous = None
        for i in self.r_peaks:
            if ((self.data[i] < 0.5 * median_all)):  # - (0.5 * std_all))):# and self.data[i] < 250):
                # if ((self.data[i] < 0.7 *median_all )):#- (0.5 * std_all))):# and self.data[i] < 250):
                # if ((self.data[i] < median_all - (0.5 * std_all))):# and self.data[i] < 250):
                # if(self.data[i] < 350):
                other_peaks.append(i)
            elif (self.data[i] < median_all and previous != None and self.data[previous] > self.data[
                i] and i - previous < 100):
                other_peaks.append(i)
            previous = i

        # eliminate non other_peaks from peaks
        top_peak_values = []
        for i in self.r_peaks:
            if i in other_peaks:
                self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == i))
            else:
                top_peak_values.append(self.data[i])

        # peaks, top_peak_values = self.get_r_peaks()

        std = np.std(top_peak_values)
        median = np.median(top_peak_values)

        if outliers_removal == True:

            for i in range(0, len(self.r_peaks)):
                # if (self.data[i] < median + (1 * std))   and (self.data[i] > median - (1* std)) :#and y[i] > 200:#
                #     new_indexes.append(i)
                down = max(0, i - 10)
                up = min(len(self.data) - 1, i + 10)

                if (self.r_peaks[i] > len(self.data)):
                    print ""

                aa_value = self.data[self.r_peaks[i]]
                aa_smaller = median + (1 * std)
                aa_bigger = median - (1 * std)

                if not ((self.data[self.r_peaks[i]] < median + (2 * std)) and (
                    self.data[self.r_peaks[i]] > median - (2 * std))):
                    outliers.append(self.r_peaks[i])
                # elif not ((self.data[down] < 0.2 * self.data[i]) and (self.data[up] < 0.2 * self.data[i])):
                #     outliers.append(self.r_peaks[i])
                elif (i < len(self.r_peaks) - 1):
                    # for i in range(0,len(self.r_peaks)-1):
                    if (self.r_peaks[i + 1] - self.r_peaks[i] < 80):
                        outliers.append(self.r_peaks[i])

            self.outliers += outliers

            outliers = sorted(outliers)
            # self.r_peaks = self.peaks

            for o in self.outliers:
                self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == o))
        else:
            # self.r_peaks = peaks
            top_peak_values = []
            for i in self.r_peaks:
                if i in other_peaks:
                    self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == i))
                else:
                    top_peak_values.append(self.data[i])

            previous = None
            distances = []
            for i in self.r_peaks:
                if previous != None:
                    distances.append(i - previous)
                previous = i

            try:
                avg_distance = np.mean(distances)
                max_distance = np.max(distances)
                min_distance = np.min(distances)
                std_distance = np.std(distances)
                # print "STD DISTANCE = " + str(std_distance)
                if (std_distance > 90):

                    stats = self.get_kmeans_stats(distances, x, 2)

                    if stats[0]['count'] >= 0.3 * len(distances) and stats[1]['count'] >= 0.3 * len(distances):

                        previous = None
                        for i in range(0, len(self.r_peaks)):
                            # if ((self.data[i] < 0.5 * median_all)):  # - (0.5 * std_all))):# and self.data[i] < 250):
                            #     # if ((self.data[i] < 0.7 *median_all )):#- (0.5 * std_all))):# and self.data[i] < 250):
                            #     # if ((self.data[i] < median_all - (0.5 * std_all))):# and self.data[i] < 250):
                            #     # if(self.data[i] < 350):
                            #     other_peaks.append(i)
                            if (previous != None and (
                                (self.r_peaks[i] - previous) < (avg_distance - 0.5 * std_distance))):
                                other_peaks.append(self.r_peaks[i])
                                previous = self.r_peaks[i]
                                i += 1
                            else:
                                previous = self.r_peaks[i]
            except:
                ""

            for i in self.r_peaks:
                if i in other_peaks:
                    self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == i))
                else:
                    top_peak_values.append(self.data[i])

        for i in range(0, len(self.r_peaks)):
            length_data = len(self.data)
            if self.r_peaks[i] == 0:
                left_of_i = 0
            else:
                left_of_i = self.r_peaks[i] - 1

            if self.r_peaks[i] == len(self.data) - 1:
                right_of_i = self.r_peaks[i]
            else:
                right_of_i = self.r_peaks[i] + 1

            value_left_of_i = self.data[left_of_i]
            value_of_i = self.data[self.r_peaks[i]]
            value_right_of_i = self.data[right_of_i]

            if value_left_of_i > value_of_i:
                self.r_peaks[i] = left_of_i
            if value_right_of_i > value_of_i:
                self.r_peaks[i] = right_of_i




    def next(self, elem, list):
        if len(list) == elem + 1:
            return elem + 1, None
        else:
            return elem + 1, list[elem + 1]

    def remove_outliers(self, outliers=None):

        self.r_peaks = np.insert(self.r_peaks, 0, [0], axis=0)
        r_peaks = copy.copy(self.r_peaks.tolist())
        if outliers == None:
            outliers = copy.copy(self.outliers)

        it_out = iter(outliers)
        it_in = iter(r_peaks)

        in_max = len(r_peaks)
        out_max = len(outliers)

        in_i = -1
        out_i = -1

        in_i, val_it = self.next(in_i, r_peaks)
        out_i, val_out = self.next(out_i, outliers)
        in_i, val_it_next = self.next(in_i, r_peaks)
        #
        # val_it = next(it_in,None)
        # val_out = next(it_out,None)
        # val_it_next = next(it_in, None)
        #


        while (True):

            if (val_it is None or val_out is None):
                break
            if (val_it <= val_out and val_out >= 0):
                # val_it_next = next(it_in, None)
                while (True):
                    if (val_it_next == None or val_out == None):
                        val_it = None
                        break
                    if (val_it_next > val_out):
                        break
                    if (val_it_next < val_out):
                        # new_indexes4.append(val_it_next)
                        val_it = val_it_next

                    in_i, val_it_next = self.next(in_i, r_peaks)
                if (val_it == None or val_out == None):
                    break

                if val_it == 0:
                    new_r = self.data[val_it_next]
                else:
                    new_r = (self.data[val_it] + self.data[val_it_next]) / 2

                self.data[val_it] = new_r

                # for i in range(val_it+1,val_it_next+1):
                self.data = np.delete(self.data, range(val_it + 1, val_it_next + 1))
                # y.remove(y[i])

                # new_indexes4.append(val_it)
                r_peaks.remove(val_it_next)
                in_i -= 1

                # self.peaks = np.delete(self.peaks, np.argwhere(self.peaks == i))

                difference = (val_it_next - val_it)
                # val_it = val_it_next
                for j in range(0, len(outliers)):
                    if (outliers[j] > val_it_next):
                        outliers[j] -= difference

                # del_index = []
                for k in range(0, len(r_peaks)):
                    # if self.r_peaks[k] == val_it_next:
                    #     del_index.append(k)
                    if (r_peaks[k] > val_it_next):
                        r_peaks[k] -= difference

                # for k in range(0, len(r_peaks)):
                #     # if self.r_peaks[k] == val_it_next:
                #     #     del_index.append(k)
                #     if (r_peaks[k] > val_it_next):
                #         r_peaks[k] -= difference
                val_it_next -= difference
                # val_out -= difference
                # self.r_peaks = np.delete(self.r_peaks, del_index)
            # val_it = val_it_next
            out_i, val_out = self.next(out_i, outliers)

        self.r_peaks = r_peaks
        # self.outliers = outliers

    def plot(self, title):
        length = len(self.data)
        x = np.linspace(0, length - 1, length)
        pyplot.figure(figsize=(10, 6))
        pplot(x, self.data, self.r_peaks)
        pyplot.title(title)

    def get_data_between_rs(self, left_r, right_r):
        new_data = []
        for i in range(left_r, right_r + 1):
            new_data.append(self.data[i])

        return new_data

    def feature_generation(self):
        feature_matrix = []
        distances = []
        r_peaks = []
        for i in range(0, len(self.r_peaks) - 1):
            try:
                distance = self.r_peaks[i + 1] - self.r_peaks[i]
                new_data = self.get_data_between_rs(self.r_peaks[i], self.r_peaks[i + 1])

                distances.append(distance)
                r_peaks.append(new_data[0])
                s_len = int(round((1.0 / 13) * distance))
                st_len = int(round((2.0 / 13) * distance)) + s_len
                t_len = int(round((4.0 / 13) * distance)) + st_len
                u_len = int(round((1.0 / 13) * distance)) + t_len
                p_len = int(round((2.0 / 13) * distance)) + u_len
                pr_len = int(round((2.0 / 13) * distance)) + p_len
                q_len = len(new_data)

                peak_distances = [0, s_len, st_len, t_len, u_len, p_len, pr_len, q_len]
                inner_features = []
                for j in range(0, len(peak_distances) - 1):
                    # for j in range(peak_distances[i], peak_distances[i+1]+1):

                    subrange = range(peak_distances[j], peak_distances[j + 1])
                    sub_data = [new_data[k] for k in subrange]
                    maxi = np.max(sub_data)
                    mini = np.min(sub_data)
                    mean = np.mean(sub_data)
                    std = np.std(sub_data)
                    inner_features.append(maxi)
                    inner_features.append(mini)
                    inner_features.append(mean)
                    inner_features.append(std)
            except:
                inner_features.append(0.0)
                inner_features.append(0.0)
                inner_features.append(0.0)
                inner_features.append(0.0)
            feature_matrix.append(inner_features)
        r_peaks.append(self.data[self.r_peaks[-1]])

        features = []
        try:
            features.append(np.max(r_peaks))
        except:
            features.append(0.0)
        try:
            features.append(np.min(r_peaks))
        except:
            features.append(0.0)
        try:
            features.append(np.mean(r_peaks))
        except:
            features.append(0.0)
        try:
            features.append(np.std(r_peaks))
        except:
            features.append(0.0)

        try:
            features.append(np.max(distances))
        except:
            features.append(0.0)
        try:
            features.append(np.min(distances))
        except:
            features.append(0.0)
        try:
            features.append(np.mean(distances))
        except:
            features.append(0.0)
        try:
            features.append(np.std(distances))
        except:
            features.append(0.0)

        feature_matrix = np.array(feature_matrix)
        try:
            matrix_length = len(feature_matrix[0])
        except:
            matrix_length = 1
        for i in range(0, matrix_length):
            # column = feature_matrix[:,i]
            try:
                features.append(np.max(feature_matrix[:, i]))
            except:
                features.append(0.0)
            try:
                features.append(np.min(feature_matrix[:, i]))
            except:
                features.append(0.0)
            try:
                features.append(np.mean(feature_matrix[:, i]))
            except:
                features.append(0.0)
            try:
                features.append(np.std(feature_matrix[:, i]))
            except:
                features.append(0.0)

        return features

# def get_r_peaks(self):
#     peaks = self.__initial_peaks__()
#
#     all_peaks_values = []
#     for i in peaks:
#         all_peaks_values.append(self.data[i])
#
#     try:
#         std_all = np.std(all_peaks_values)
#         median_all = np.median(all_peaks_values)
#         max_all = np.max(all_peaks_values)
#     except:
#         std_all = 0
#         median_all = 0
#         max_all = 0
#
#     x = np.array([all_peaks_values]).T
#
#     try:
#         # k=1
#         r_k = None
#         r_k_center = 0
#         std_k = 99
#         for k in range(1, 4):
#             kmeans = KMeans(n_clusters=k)
#             prediction = kmeans.fit_predict(x)
#
#             stats = {}
#             for i in range(0, len(all_peaks_values)):
#                 if prediction[i] not in stats:
#                     stats[prediction[i]] = {}
#                     stats[prediction[i]]["values"] = []
#                     stats[prediction[i]]["count"] = 0
#                 stats[prediction[i]]["count"] += 1
#                 stats[prediction[i]]["values"].append(all_peaks_values[i])
#
#             for key, value in stats.iteritems():
#                 stats[key]["std"] = np.std(value["values"])
#                 stats[key]["mean"] = kmeans.cluster_centers_[key]
#                 std_perc = stats[key]["std"] / stats[key]["mean"]
#                 a_first_left = float(stats[key]["count"]) / (len(all_peaks_values))
#                 a_first_right = 1.0 / (2 * k)
#                 a_first = (float(stats[key]["count"]) / (len(all_peaks_values)) > (1.0 / (2 * k)))
#                 a_second = stats[key]["mean"] > r_k_center
#                 a_third = stats[key]["std"] / stats[key]["mean"] < 1.2 * std_k
#
#                 print"Her"
#                 if (float(stats[key]["count"]) / (len(all_peaks_values)) > (1.0 / (k + 1))) and \
#                                 stats[key]["mean"] > r_k_center and stats[key]["std"] / stats[key][
#                     "mean"] < 1.2 * std_k:
#                     r_k = key
#                     median_all = stats[key]["mean"]
#                     std_k = stats[key]["std"] / stats[key]["mean"]
#
#         new_r_peaks = []
#
#         # for i in range(0,len(prediction)):
#         #     if prediction[i] == r_k:
#         #         new_r_peaks.append(all_peaks_values[i])
#         # self.r_peaks = new_r_peaks
#
#
#
#
#
#     except:
#         print"to less"
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     #
#     outliers = [0]
#     # r_peaks = []
#     other_peaks = []
#
#     # find NOT outliers but not R Peaks:
#
#     # for i in peaks:
#     #     if ((self.data[i] < 0.5 * max_all)):  # and self.data[i] < 250):
#     #         # if ((self.data[i] < max_all - (1 * std_all))):  # and self.data[i] < 250):
#     #         # if(self.data[i] < 350):
#     #         other_peaks.append(i)
#
#     for i in peaks:
#         if ((self.data[i] < 0.5 * median_all)):  # - (0.5 * std_all))):# and self.data[i] < 250):
#             # if ((self.data[i] < 0.7 *median_all )):#- (0.5 * std_all))):# and self.data[i] < 250):
#             # if ((self.data[i] < median_all - (0.5 * std_all))):# and self.data[i] < 250):
#             # if(self.data[i] < 350):
#             other_peaks.append(i)
#         # elif (previous != None and i - previous < 50):
#         #     other_peaks.append(i)
#
#
#         previous = i
#
#     # eliminate non other_peaks from peaks
#     self.r_peaks = peaks
#     top_peak_values = []
#     for i in peaks:
#         if i in other_peaks:
#             self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == i))
#         else:
#             top_peak_values.append(self.data[i])
#
#     previous = None
#     distances = []
#     for i in self.peaks:
#         if previous != None:
#             distances.append(i - previous)
#             previous = i
#
#     avg_distance = np.mean(distances)
#
#     previous = None
#     for i in self.r_peaks:
#         # if ((self.data[i] < 0.5 * median_all)):  # - (0.5 * std_all))):# and self.data[i] < 250):
#         #     # if ((self.data[i] < 0.7 *median_all )):#- (0.5 * std_all))):# and self.data[i] < 250):
#         #     # if ((self.data[i] < median_all - (0.5 * std_all))):# and self.data[i] < 250):
#         #     # if(self.data[i] < 350):
#         #     other_peaks.append(i)
#         if (previous != None and i - previous < avg_distance):
#             other_peaks.append(i)
#
#         previous = i
#
#         # eliminate non other_peaks from peaks
#     self.r_peaks = peaks
#     top_peak_values = []
#     for i in peaks:
#         if i in other_peaks:
#             self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == i))
#         else:
#             top_peak_values.append(self.data[i])
#
#     return peaks, top_peak_values
