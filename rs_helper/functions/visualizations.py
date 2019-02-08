import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rs_helper.classes import EmbeddingModel
from typing import *
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity


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

    df = pd.DataFrame.from_dict({"text": text, "label": labels})
    df["embedding"] = df["text"].apply(lambda x: embedding_model.inference(word_tokenize(x)))
    # df["embedding"] = df["text"].apply(lambda x: embedding_model.inference(word_tokenize(x))[0])
    # PCA
    # print(df["embedding"])
    pca = PCA(n_components=2)
    matrix = pca.fit_transform(df["embedding"].apply(lambda x: x[0]).tolist())

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
    legend = plt.legend(handles=labels)

    if save_dir is not None and os.path.isdir(save_dir):
        plt.savefig(os.path.join(save_dir, "scatter.png"))

    # TODO Remove this later
    plt.savefig(os.path.join("notebooks/output_scatter", "{}.png".format(str(ax_title))))
    return ax_1, df


def similarity_matrix(messages: list, vectors: list, name: str, save_path: str = "."):
    similarity_matrix_data = cosine_similarity(vectors, vectors)
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.set(font_scale=1.5)
    heatmap = sns.heatmap(
        similarity_matrix_data,
        xticklabels=messages,
        yticklabels=messages,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        ax=ax)
    heatmap.set_xticklabels(messages, rotation=90)
    heatmap.set_title(name)
    plt.savefig(os.path.join(save_path, "similarity_matrix.svg"))
    plt.close()
