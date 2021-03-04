import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.ma import sort

import download as dl
import model2 as mod2


def plot_scores():
    all_data = pd.DataFrame()
    for ye in range(2020, 2021):
        for le in ['F1']:
            all_data = all_data.append(dl.download_scores(ye, le), ignore_index=True, sort=False)

    scores = np.zeros((np.max(all_data.FTAG) + 1, np.max(all_data.FTHG) + 1))
    for i in range(0, np.max(all_data.FTHG)+1):
        for j in range(0, np.max(all_data.FTAG)+1):
            scores[j, i] = len(all_data[(all_data.FTHG == i) & (all_data.FTAG == j)].index)
    fig, ax = plt.subplots()
    im = ax.imshow(scores)
    for i in range(np.max(all_data.FTAG) + 1):
        for j in range(np.max(all_data.FTHG) + 1):
            text = ax.text(j, i, "{:.2f}".format(scores[i, j]),
                           ha="center", va="center", color="w")
    ax.set_title("Log nb scores in French League 1")
    fig.tight_layout()
    plt.show()


def plot_accuracy():
    acc_traces = mod2.get_model()[1]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')
    ax1.plot(acc_traces['nb_epochs'], acc_traces['accuracy_train'])
    ax1.plot(acc_traces['nb_epochs'], acc_traces['accuracy_test'])
    ax2.plot(acc_traces['nb_epochs'], acc_traces['loss_train'])
    ax2.plot(acc_traces['nb_epochs'], acc_traces['loss_test'])


#plot_scores()
plot_accuracy()
