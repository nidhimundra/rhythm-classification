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
        minus = 0
        plus = 0
        for i in data:
            if i > 200:
                plus += 1
            if i < -200:
                minus += 1
        if minus > plus:
            return -data
        else:
            return data

    def __initial_peaks__(self):
        return peakutils.indexes(self.data, thres=0.46, min_dist=30)

    def get_r_peaks(self):
        peaks = self.__initial_peaks__()
        all_peaks_values = []
        for i in peaks:
            all_peaks_values.append(self.data[i])

        try:
            std_all = np.std(all_peaks_values)
            median_all = np.median(all_peaks_values)
            max_all = np.max(all_peaks_values)
        except:
            std_all = 0
            median_all = 0
            max_all = 0
        #
        outliers = [0]
        # r_peaks = []
        other_peaks = []

        # find NOT outliers but not R Peaks:

        for i in peaks:
            if ((self.data[i] < 0.5 * max_all)):  # and self.data[i] < 250):
                # if ((self.data[i] < max_all - (1 * std_all))):  # and self.data[i] < 250):
                # if(self.data[i] < 350):
                other_peaks.append(i)

        # eliminate non other_peaks from peaks
        self.r_peaks = peaks
        top_peak_values = []
        for i in peaks:
            if i in other_peaks:
                self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == i))
            else:
                top_peak_values.append(self.data[i])

        return peaks, top_peak_values

    def find_outliers(self):

        # data = copy.deepcopy(self.data)
        while True:
            # peaks = copy.deepcopy(self.r_peaks)
            self.r_peaks = self.__initial_peaks__()

            all_peaks_values = []
            for i in self.r_peaks:
                all_peaks_values.append(self.data[i])

            x = np.array([all_peaks_values]).T
            std_all = np.std(all_peaks_values)
            median_all = np.median(all_peaks_values)
            try:
                # k=1
                for k in range(1, 4):
                    kmeans = KMeans(n_clusters=k)
                    prediction = kmeans.fit_predict(x)

                    stats = {}
                    for i in range(0, len(all_peaks_values)):
                        if prediction[i] not in stats:
                            stats[prediction[i]] = {}
                            stats[prediction[i]]["values"] = []
                            stats[prediction[i]]["count"] = 0
                        stats[prediction[i]]["count"] += 1
                        stats[prediction[i]]["values"].append(all_peaks_values[i])

                    r_k = None
                    r_k_center = 0

                    for key, value in stats.iteritems():
                        stats[key]["std"] = np.std(value["values"])
                        stats[key]["mean"] = kmeans.cluster_centers_[key]
                        if (stats[key]["count"] / (len(all_peaks_values)) > 1 / (k + 1)) and stats[key][
                            "mean"] > r_k_center:
                            r_k = key
                            median_all = stats[key]["mean"]

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

            print len(outliers)
            if len(outliers) <= 2:
                self.get_r_peaks()
                break
            else:
                self.outliers += outliers

            outliers = sorted(outliers)
            # self.r_peaks = self.peaks

            for o in self.outliers:
                self.r_peaks = np.delete(self.r_peaks, np.argwhere(self.r_peaks == o))

            self.remove_outliers(outliers)
            break




            # for n in range(0, len(other_peaks)):
            #     np.delete(self.r_peaks, [n])

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
