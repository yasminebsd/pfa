import itertools
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import settings
from settings import classes


def display_image_with_multiple_layers(image):
    for i in range(0, image.shape[2]):
        img = image[:, :, i]
        plt.imshow(img, alpha=.3)
    plt.show()


def display_data_diff(image, train_image, test_image):
    fig = plt.figure(figsize=(10, 3))
    ax = []

    ax.append(fig.add_subplot(1, 3, 1))
    ax[-1].set_title("data")
    plt.imshow(image[:, :, 0])

    ax.append(fig.add_subplot(1, 3, 2))
    ax[-1].set_title("train")
    plt.imshow(train_image[:, :, 0])

    ax.append(fig.add_subplot(1, 3, 3))
    ax[-1].set_title("test")
    plt.imshow(test_image[:, :, 0])

    plt.show()


def display_one_dim_image(img):
    plt.imshow(img)
    plt.show()


def displayClassTable(number_of_list, title="                 Nombre d'échantillons"):
    print("\n+------------ Tableau d'échantillons ---------------+")
    lenth = len(number_of_list)
    column1 = range(1, lenth + 1)
    table = {'Class#': column1, title: number_of_list}
    table_df = pd.DataFrame(table).to_string(index=False)
    print(table_df)
    print("+------------ Tableau d'échantillons ---------------+")

def groundTruthVisualise(data):
    from matplotlib.pyplot import show
    # set_cmap('tab20b')

    cmap = plt.get_cmap('Paired', np.max(data) - np.min(data) + 1)

    mat = plt.matshow(data,
                      cmap=cmap,
                      vmin=np.min(data) - .5,
                      vmax=np.max(data) + .5)

    cb = plt.colorbar(mat,
                      ticks=np.arange(np.min(data), np.max(data) + 1))
    cb.ax.set_yticklabels(np.array(settings.classes))

    show()


def groundTruthDifferenceVisualise(data, predictedData):
    from matplotlib.pyplot import imshow, show, set_cmap
    fig = plt.figure(figsize=(20, 10))
    ax = []

    colors = ['black', 'white']
    bounds = [0, 1]

    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # cmap = plt.get_cmap('Paired', np.max(data) - np.min(data) + 1)


    ax.append(fig.add_subplot(1, 2, 1))
    ax[-1].set_title("data")
    mat = plt.imshow(data,
                      cmap=cmap,
                      vmin=np.min(data) - .5,
                      vmax=np.max(data) + .5)

    ax.append(fig.add_subplot(1, 2, 2))
    ax[-1].set_title("predicted data")
    mat = plt.imshow(predictedData,
                      cmap=cmap,
                      vmin=np.min(data) - .5,
                      vmax=np.max(data) + .5)

    cb = plt.colorbar(mat,
                      ticks=np.arange(np.min(data), np.max(data) + 1))
    cb.ax.set_yticklabels(np.array(settings.classes))



    show()


def display_confusion_matrix(data, predictedData):
    a1 = np.array(data).flatten()
    a2 = np.array(predictedData).flatten()
    cm = confusion_matrix(a1, a2)
    plot_confusion_matrix(cm, classes)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
