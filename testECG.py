import csv
import os

import matplotlib.pyplot as plt
import scipy.io


def read_lable_dict():
    with open('training2017/REFERENCE.csv', mode='r') as infile:
        reader = csv.reader(infile)
        # with open('coors_new.csv', mode='w') as outfile:
        # writer = csv.writer(outfile)
        mydict = {rows[0]: rows[1] for rows in reader}
    return mydict


def plot_line_graph(arrays, labels, title_img):
    # label = label_dict[title_img]
    width = arrays[0].size
    fig = plt.figure(num=None, figsize=(width / 200, 3), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    # ticks = np.arange(0.0, 0.5, 0.02)
    for i in range(len(arrays)):
        array = arrays[i]
        index = range(len(array))
        values = array
        ax.plot(values)  # , colors[i])
        # ax.set_xlabel(tuning_parameter)
        ax.set_ylabel('RMSE Score')
    plt.title(title_img)
    plt.legend(labels, loc='lower right')

    plt.savefig("plots/" + title_img + "_" + label + ".pdf")


for filename in os.listdir('training2017'):
    if filename.endswith('.mat'):
        name = filename[:-4]
        label_dict = read_lable_dict()
        label = label_dict[name]
        mat1 = scipy.io.loadmat('training2017/' + filename)
        plot_line_graph([mat1['val'][0]], [label], name)

#
# mat2 = scipy.io.loadmat('A00002.mat')
# mat3 = scipy.io.loadmat('A00003.mat')
# mat4 = scipy.io.loadmat('A00004.mat')
# mat5 = scipy.io.loadmat('A00005.mat')
# mat6 = scipy.io.loadmat('A00006.mat')


print
"done"
