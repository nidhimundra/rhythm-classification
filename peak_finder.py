"""
 Created by Nidhi Mundra on 25/04/17.
"""

import copy
import numpy as np
from sklearn.cluster import KMeans
import peakutils


class PeakFinder:
    def __init__(self, data):
        """
        Initialize the wave data, Find R peaks and Remove outliers 
        :param data: Wave data points 
        """

        self.data = data

        # Flip the wave if negative peaks are more
        self.__flip_data__()

        # Get all peaks
        self.__initial_peaks__()

        # Initialize r_peaks array
        self.r_peaks = copy.copy(self.peaks)

        # Initialize outlier array
        self.outliers = []

        # Detect and remove outliers
        self.__outlier_removal__()

    def __flip_data__(self):
        """
        Flip the wave if the negative data points are more than the positive data points
        """
        negative_count = 0
        positive_count = 0

        for i in self.data:
            if i > 200:
                positive_count += 1
            elif i < -200:
                negative_count += 1

        # Flip the wave if negative_count is more
        if negative_count > positive_count:
            self.data = -self.data

    def __initial_peaks__(self):

        """
        Detect peaks using peakutils
        """
        self.peaks = peakutils.indexes(self.data, thres=0.46, min_dist=30)

    def __outlier_removal__(self):
        """
        Detect R peaks and remove outliers adn restitch the wave after outlier removal
        """
        self.__find_r_peaks__(outliers_removal=True)
        self.__remove_outliers__(self.outliers)
        self.__find_r_peaks__(outliers_removal=False)

    def __get_cluster_stats__(self, X, k):
        """
        Cluster the peaks to find the moments (count, mean and standard deviation) of each cluster
        :param X: Data to be clustered
        :param k: No. of clusters
        :return: Statistical data of each cluster
        """
        kmeans = KMeans(n_clusters=k)
        prediction = kmeans.fit_predict(X)
        stats = {}
        for i in range(0, len(self.data)):
            if prediction[i] not in stats:
                stats[prediction[i]] = {}
                stats[prediction[i]]["values"] = []
                stats[prediction[i]]["count"] = 0
            stats[prediction[i]]["count"] += 1
            stats[prediction[i]]["values"].append(self.data[i])
        for key, value in stats.iteritems():
            stats[key]["std"] = np.std(value["values"])
            stats[key]["mean"] = kmeans.cluster_centers_[key]

        return stats

    def __find_r_peaks__(self, outliers_removal=False):
        """
        Detect R Peaks in the wave data
        :param outliers_removal: True if outlier removal to be done, False otherwise
        """

        all_peaks_values = []
        for i in self.r_peaks:
            all_peaks_values.append(self.data[i])

        x = np.array([all_peaks_values]).T
        median_all = np.median(all_peaks_values)
        try:
            r_k_center = 0
            std_k = 99
            for k in range(1, 4):
                stats = self.__get_cluster_stats__(all_peaks_values, x, k)
                for key, value in stats.iteritems():
                    if (float(stats[key]["count"]) / (len(all_peaks_values)) > (1.0 / (2 * k))) and \
                                    stats[key]["mean"] > r_k_center and stats[key]["std"] / stats[key][
                        "mean"] < 1.2 * std_k:
                        median_all = stats[key]["mean"]
                        std_k = stats[key]["std"] / stats[key]["mean"]


        except:
            ""

        outliers = [0]
        other_peaks = []

        # find NOT outliers but not R Peaks:
        previous = None
        for i in self.r_peaks:
            if self.data[i] < 0.5 * median_all:
                other_peaks.append(i)
            elif (self.data[i] < median_all and previous is not None and self.data[previous] > self.data[
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

        std = np.std(top_peak_values)
        median = np.median(top_peak_values)

        if outliers_removal:

            for i in range(0, len(self.r_peaks)):

                if self.r_peaks[i] > len(self.data):
                    print ""

                if not ((self.data[self.r_peaks[i]] < median + (2 * std)) and (
                            self.data[self.r_peaks[i]] > median - (2 * std))):
                    outliers.append(self.r_peaks[i])
                elif i < len(self.r_peaks) - 1:
                    if self.r_peaks[i + 1] - self.r_peaks[i] < 80:
                        outliers.append(self.r_peaks[i])

            self.outliers += outliers

            for o in self.outliers:
                self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == o))
        else:
            top_peak_values = []
            for i in self.r_peaks:
                if i in other_peaks:
                    self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == i))
                else:
                    top_peak_values.append(self.data[i])

            previous = None
            distances = []
            for i in self.r_peaks:
                if previous is not None:
                    distances.append(i - previous)
                previous = i

            try:
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)

                if std_distance > 90:

                    stats = self.__get_cluster_stats__(distances, x, 2)

                    if stats[0]['count'] >= 0.3 * len(distances) and stats[1]['count'] >= 0.3 * len(distances):

                        previous = None
                        for i in range(0, len(self.r_peaks)):
                            if (previous is not None and (
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

    def __next__(self, element, list_items):
        """
        Return the next element of the string 
        :param element: Current element index
        :param list_items: Array of all the elements
        :return: Index and value of the next item.
         Returns None in case current element was the last element of the list items
        """
        if len(list_items) == element + 1:
            return element + 1, None
        else:
            return element + 1, list_items[element + 1]

    def __remove_outliers__(self, outliers=None):
        """
        Remove Outliers from the detected peaks
        :param outliers: Array of outliers
        """
        self.outliers = []
        self.r_peaks = copy.copy(self.peaks)
        self.r_peaks = np.insert(self.r_peaks, 0, [0], axis=0)
        r_peaks = copy.copy(self.r_peaks.tolist())
        if outliers is None:
            outliers = copy.copy(self.outliers)

        in_i = -1
        out_i = -1

        in_i, val_it = self.__next__(in_i, r_peaks)
        out_i, val_out = self.__next__(out_i, outliers)
        in_i, val_it_next = self.__next__(in_i, r_peaks)

        while True:
            if val_it is None or val_out is None:
                break
            if val_it <= val_out and val_out >= 0:
                while True:
                    if val_it_next is None or val_out is None:
                        val_it = None
                        break
                    if val_it_next > val_out:
                        break
                    if val_it_next < val_out:
                        val_it = val_it_next

                    in_i, val_it_next = self.__next__(in_i, r_peaks)
                if val_it is None or val_out is None:
                    break

                if val_it == 0:
                    new_r = self.data[val_it_next]
                else:
                    new_r = (self.data[val_it] + self.data[val_it_next]) / 2

                self.data[val_it] = new_r

                self.data = np.delete(self.data, range(val_it + 1, val_it_next + 1))
                r_peaks.remove(val_it_next)
                in_i -= 1

                difference = (val_it_next - val_it)
                for j in range(0, len(outliers)):
                    if outliers[j] > val_it_next:
                        outliers[j] -= difference

                for k in range(0, len(r_peaks)):
                    if r_peaks[k] > val_it_next:
                        r_peaks[k] -= difference

                val_it_next -= difference
            out_i, val_out = self.__next__(out_i, outliers)

        self.r_peaks = r_peaks

    def get_peaks_data(self):
        """
        Get all the R Peaks and the Wave data after outlier removal
        :return: R peak points and the transformed wave data
        """
        return [self.r_peaks, self.data]
