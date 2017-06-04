"""
 Created by Jonas Pfeiffer on 26/04/17.
"""

import copy

import numpy as np
from matplotlib import pyplot
from sklearn.cluster import KMeans

import peakutils
from peakutils.plot import plot as pplot


class PeakFinder:
    def __init__(self, data, outliers):
        """
        Initialize the wave data, Find R peaks and Remove outliers
         
        :param data: Wave data points 
        """

        self.data = data

        # Flip the wave if negative peaks are more
        # self.__flip_data__()

        # Initialize outlier array
        self.outliers = outliers

        # Get all peaks
        self.__initial_peaks__()

        # Initialize r_peaks array
        self.r_peaks = copy.copy(self.peaks)

        # Detect and remove outliers
        self.__outlier_removal__()

    def __flip_data__(self):
        """
        Flip the wave if the negative data points are more than the positive data points
        """

        # Initialize counters
        negative_count = 0
        positive_count = 0

        for i in self.data:

            # Increment the positive count if the data point is greater than 200
            if i > 200:
                positive_count += 1

            # Increment the negative count if the data point is lesser than 200
            elif i < -200:
                negative_count += 1

        # Flip the wave if negative_count is more
        if negative_count > positive_count:
            self.data = -self.data

    def __initial_peaks__(self):
        """
        Detect peaks using peakutils library
        """
        self.peaks = peakutils.indexes(self.data, thres=0.46, min_dist=30)

    def __outlier_removal__(self):
        """
        Detect R peaks and remove outliers adn restitch the wave after outlier removal
        """

        # Find all the R peaks in the wave
        self.__find_r_peaks__(outliers_removal=True)

        # Remove outliers and transform the data
        self.__remove_outliers__(self.outliers)

        # Find outliers from the transformed data
        self.__find_r_peaks__(outliers_removal=False)

    def __get_cluster_stats__(self, all_peak_values, X, k):
        """
        Cluster the peaks to find the moments (count, mean and standard deviation) of each cluster
        
        :param X: Data to be clustered
        
        :param k: No. of clusters
        
        :return: Statistical data of each cluster
        """

        # Initialize output
        stats = {}

        # Cluster the data into k clusters
        kmeans = KMeans(n_clusters=k)
        prediction = kmeans.fit_predict(X)

        for i in range(0, len(all_peak_values)):

            # Store the values and count of elements of each cluster
            if prediction[i] not in stats:
                # Initialize on first occurence
                stats[prediction[i]] = {}
                stats[prediction[i]]["values"] = []
                stats[prediction[i]]["count"] = 0

            # Increment the count and store the value
            stats[prediction[i]]["count"] += 1
            stats[prediction[i]]["values"].append(all_peak_values[i])

        for key, value in stats.iteritems():
            # Compute standard deviation and mean of the values and cluster centers
            stats[key]["std"] = np.std(value["values"])
            stats[key]["mean"] = kmeans.cluster_centers_[key]

        return stats

    def __find_r_peaks__(self, outliers_removal=False):
        """
        Detect R Peaks in the wave data
        
        :param outliers_removal: True if outlier removal to be done, False otherwise
        """

        # Initialize the output
        all_peaks_values = []


        # Store the data value of all the R peak indices
        for i in range(0, len(self.r_peaks)):
            if self.r_peaks[i] >= len(self.data):
                # TODO this is a hack. File A03017 and maybe others after this somehow have peaks outside of the range. Find out why...
                self.r_peaks[i] = len(self.data) - 1
            all_peaks_values.append(self.data[self.r_peaks[i]])

        # Transpose the data to perform clustering of each peak
        x = np.array([all_peaks_values]).T

        # Compute median of all the peak values
        median_all = np.median(all_peaks_values)

        try:

            # Initialize best R cluster and its standard deviation
            best_r_cluster = 0
            std_cluster = 99

            for k in range(1, 4):
                """
                Cluster the peaks data to predict the R peaks correctly.
                The cluster with medium number of data points and
                are higher than most of the data points is selected
                as the one with most accurate R peaks.
                """

                # Compute stats for the clusters
                stats = self.__get_cluster_stats__(all_peaks_values, x, k)

                for key, value in stats.iteritems():

                    # if the counts of the intermediate cluster is greater than 1/2k and the avg value in the cluster is
                    # higher than the one previously found and the standard deviation is at least 20% better then
                    # use the new cluster mean as the median_all
                    if (float(stats[key]["count"]) / (len(all_peaks_values)) > (1.0 / (2 * k))) and \
                                    stats[key]["mean"] > best_r_cluster and \
                                            stats[key]["std"] / stats[key]["mean"] < 1.2 * std_cluster:
                        # Compute the median and standard deviation of the data
                        median_all = stats[key]["mean"]
                        std_cluster = stats[key]["std"] / stats[key]["mean"]


        except:
            print "to less peaks to cluster"

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

        # if outlier_removal has been turned on, remove the outliers
        if outliers_removal:

            for i in range(0, len(self.r_peaks)):
                # if the current peak's value is not in the range of median in a threshold of 2* standard deviation then add the
                # peak to outliers
                if not ((self.data[self.r_peaks[i]] < median + (2 * std)) and (
                            self.data[self.r_peaks[i]] > median - (2 * std))):
                    outliers.append(self.r_peaks[i])
                # if the peak's values do not decrease in the next 80 indices very fast,
                # add the peak to outliers
                elif i < len(self.r_peaks) - 1:
                    if self.r_peaks[i + 1] - self.r_peaks[i] < 80:
                        outliers.append(self.r_peaks[i])

            # append to outliers
            self.outliers += outliers

            # remove outliers from r_peaks
            for o in self.outliers:
                self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == o))


        # if not outlier removal, find the R-Peaks
        else:

            # Same as above but without assuming that there are still outliers
            top_peak_values = []
            for i in self.r_peaks:
                if i in other_peaks:
                    self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == i))
                else:
                    top_peak_values.append(self.data[i])

            # calculate the distance between two peaks
            previous = None
            distances = []
            for i in self.r_peaks:
                if previous is not None:
                    distances.append(i - previous)
                previous = i

            try:

                # calculate the std of the distances
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)

                # If the STD is at least 90 we will cluster the distances
                # this is done because often the T-Peak has a similar hight compared to the R-Peak
                # Therefore the T-Peak is classified as a R-Peak. If find two clusters of distances
                # that have a very low std, we can assume that the high std of distances is only due
                # to the fact that we have a short distance between the R-T peaks but a high distance
                # between the T-R peaks. We therefore delete the second (T-) Peak from R_Peaks

                if std_distance > 90:

                    # cluster the distances into two clusters
                    stats = self.__get_cluster_stats__(distances, x, 2)

                    # if the clusters are evenly distributed then we assume a R-T peak case
                    if stats[0]['count'] >= 0.3 * len(distances) and stats[1]['count'] >= 0.3 * len(distances):

                        # derive which one is the R-Peak and which one is the T-Peak and add it to other_peaks
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

            # eliminate other_peaks from r_peaks
            for i in self.r_peaks:
                if i in other_peaks:
                    self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == i))
                else:
                    top_peak_values.append(self.data[i])

        # This is a hack. sometimes the found peak is off by one to the left or right. This just checks which one is higher
        # and resets the r_peak to that value.
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
        Remove Outliers the dataset and recalculate the peaks.
        Logic is if we find an outlier, we delete all values between two found R-Peaks.
        This way a whole chunk of the ECG is immediately deleted.
        Problem is that if we delete a chunk, each peak that comes after the chunk, needs
        to be updated by the amount of datapoints that were deleted
        
        :param outliers: Array of outliers
        """

        self.r_peaks = np.insert(self.r_peaks, 0, [0], axis=0)
        r_peaks = copy.copy(self.r_peaks.tolist())
        if outliers is None:
            outliers = copy.copy(self.outliers)

        in_i = -1
        out_i = -1

        in_i, val_it = self.__next__(in_i, r_peaks)
        out_i, val_out = self.__next__(out_i, outliers)
        in_i, val_it_next = self.__next__(in_i, r_peaks)

        # This algorithm loops through all the outliers and finds the R-Peak before
        # the outlier and the next R-Peak and deletes all values from the ECG
        while True:

            # if there is no next values, break
            if val_it is None or val_out is None:
                break

            if val_it <= val_out and val_out >= 0:

                # Loop until a outlier , an R lowe and an R higher than the outlier is found
                while True:

                    # if either no more outliers, or no more no more R-Peaks, break
                    if val_it_next is None or val_out is None:
                        val_it = None
                        break

                    # if the next R Peak is higher than than the outlier break
                    if val_it_next > val_out:
                        break

                    # if the next R Peak is still smaller than the outlier, update the lower R-Peak
                    if val_it_next < val_out:
                        val_it = val_it_next

                    # get the next R-Peak
                    in_i, val_it_next = self.__next__(in_i, r_peaks)

                # if either no more outliers, or no more no more R-Peaks, break
                if val_it is None or val_out is None:
                    break

                # initialize if we are at the first R-Peak and its index 0, then we will take the new R-Peak as the
                # next R-Peak values
                if val_it == 0:
                    new_r = self.data[val_it_next]

                # Else merge the two R-Peaks as an average of the previous and next hight
                else:
                    new_r = (self.data[val_it] + self.data[val_it_next]) / 2

                # update the R-peak value in the dataset
                self.data[val_it] = new_r

                # delete the range of one R-Peak to the next from the dataset
                self.data = np.delete(self.data, range(val_it + 1, val_it_next + 1))

                # Remove the next R peak from the R_peaks and step one back for indexes of R-Peaks
                # because there might be other outliers in this interval.
                r_peaks.remove(val_it_next)
                in_i -= 1

                # update the indices of each outlier
                difference = (val_it_next - val_it)
                for j in range(0, len(outliers)):
                    if outliers[j] > val_it_next:
                        outliers[j] -= difference

                # update the indices of each R-Peak
                for k in range(0, len(r_peaks)):
                    if r_peaks[k] > val_it_next:
                        r_peaks[k] -= difference

                # update the already called val_it_next
                val_it_next -= difference

            # get the next outlier
            out_i, val_out = self.__next__(out_i, outliers)

        index = -1
        new_r_peaks = []
        for one_peak in r_peaks:

            if index < one_peak:
                index = one_peak
                new_r_peaks.append(one_peak)

        self.r_peaks = new_r_peaks

    def get_peaks_data(self):
        """
        Get all the R Peaks and the Wave data after outlier removal
        
        :return: R peak points and the transformed wave data
        """
        return [self.r_peaks, self.data]

    def plot(self, title, data=None):
        """
        Plot the ECG wave using self.data

        :param title: The title of the waveform
        """

        if data == None:
            data = self.data

        # Data to be plotted
        y = np.linspace(0, len(data) - 1, len(data))

        pyplot.figure(figsize=(10, 6))

        # Plot using pplot library
        pplot(y, data, self.r_peaks)

        pyplot.title(title)


def plot(title, data, peaks=[0]):
    """
    Plot the ECG wave using self.data

    :param title: The title of the waveform
    """

    # Data to be plotted
    pyplot.close("all")
    y = np.linspace(0, len(data) - 1, len(data))

    pyplot.figure(figsize=(10, 6))

    # Plot using pplot library
    pplot(y, data, peaks)

    pyplot.title(title)
