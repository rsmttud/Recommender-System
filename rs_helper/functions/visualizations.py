import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rs_helper.classes import EmbeddingModel, DAN
from typing import *
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.colors


# TODO Obsolete!
def kd_plot(embedding_model: EmbeddingModel,
            text: List[str],
            labels: List[Any],
            save_dir: str = None,
            fig_title: str = "",
            ax_title: str = "",
            cmaps: List[str] = ["Reds", "YlOrBr", "Blues"]):
    if len(labels) != len(text):
        raise ValueError("List of text and labels need to have the same len")
    if len(cmaps) < len(set(labels)):
        raise ValueError("You need to provide the at least the same amount of cmaps as classes in the data: {}".format(
            len(set(text))))

    df = pd.DataFrame.from_dict({"text": text, "label": labels})
    df["embedding"] = df["text"].apply(lambda x: embedding_model.inference(word_tokenize(x))[0])
    # PCA
    # print(df["embedding"])
    pca = PCA(n_components=2)
    matrix = pca.fit_transform(df["embedding"].tolist())
    df["x"] = matrix[:, 0]
    df["y"] = matrix[:, 1]
    # Plot
    groups = df.groupby("label")

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(fig_title)

    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1.set_title(ax_title)

    labels = []

    for i, (class_name, _df) in enumerate(groups):
        ax = sns.kdeplot(_df["x"], _df["y"], shade=True, cmap=cmaps[i], shade_lowest=False, ax=ax_1)
        color_from_cmap = plt.cm.get_cmap(cmaps[i])(0.5)
        labels.append(mpatches.Patch(color=color_from_cmap, label=class_name))
    legend = plt.legend(handles=labels)

    if os.path.isdir(save_dir):
        plt.savefig(os.path.join(save_dir, "kd_plot.png"))
    return ax_1


def scatter_plot(embedding_model: EmbeddingModel,
                 text: List[str],
                 labels: List[Any],
                 save_dir: str = None,
                 fig_title: str = "",
                 ax_title: str = "",
                 colors: List[str] = ["#EF4836", "#FDB50A", "#00678F"]):
    if len(labels) != len(text):
        raise ValueError("List of text and labels need to have the same len")
    if len(colors) < len(set(labels)):
        raise ValueError("You need to provide the at least the same amount of cmaps as classes in the data: {}".format(
            len(set(text))))

    df = pd.DataFrame.from_dict({"text": [word_tokenize(x) for x in text], "label": labels})
    if isinstance(embedding_model, DAN):
        df["embedding"] = embedding_model.inference_batches(df["text"].tolist())
    else:
        df["embedding"] = df["text"].apply(lambda x: embedding_model.inference(x))
        df["embedding"] = df["embedding"].apply(lambda x: np.mean(x, axis=0))
    # df["embedding"] = df["text"].apply(lambda x: embedding_model.inference(word_tokenize(x))[0])
    # PCA
    # print(df["embedding"])
    pca = PCA(n_components=2)

    if isinstance(embedding_model, DAN):
        matrix = pca.fit_transform(df["embedding"].apply(lambda x: x[0]).tolist())

    else:
        matrix = pca.fit_transform(df["embedding"].tolist())
    explained_variance = pca.explained_variance_ratio_.cumsum()

    df["x"] = matrix[:, 0]
    df["y"] = matrix[:, 1]
    # Plot
    groups = df.groupby("label")

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(fig_title)

    ax_1 = fig.add_subplot(1, 1, 1)
    ax_1.set_title(ax_title)

    labels = []

    for i, (class_name, _df) in enumerate(groups):
        ax_1.scatter(_df["x"], _df["y"], c=colors[i], alpha=0.4)
        labels.append(mpatches.Patch(color=colors[i], label=class_name))
    # line1 = Line2D(range(10), range(10), marker='o', color="goldenrod")
    print(str(explained_variance))
    legend = plt.legend(handles=labels)

    if save_dir is not None and os.path.isdir(save_dir):
        plt.savefig(os.path.join(save_dir, "scatter.png"))
        plt.savefig(os.path.join(save_dir, "scatter.svg"))

    # TODO Remove this later

    #plt.savefig(os.path.join("output_scatter/png/", "{}.png".format(str(ax_title))))
    #plt.savefig(os.path.join("output_scatter/svg/", "{}.svg".format(str(ax_title))))
    return ax_1


def confusion_matrix_plot(y_true: List[int], y_pred: List[int],title:str, labels=None, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
       cm =  cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Color Map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
                                                               ["#009FDB", "#00719C", "#00678F", "#005575", "#00394F"])

    # Heatmap
    ax = plt.axes()
    ax.set_title(title + "\n" + str(accuracy_score(y_true, y_pred)))
    #ax.set_title(title)
    heatmap_svm = sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, ax=ax, cmap=cmap)
    plt.ylabel('Y_True')
    plt.xlabel('Y_Pred')

    return ax
