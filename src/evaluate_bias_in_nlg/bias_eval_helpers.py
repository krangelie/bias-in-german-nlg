import os

import pandas as pd
import seaborn as sns
from matplotlib import patches as mpatches, pyplot as plt

from src import constants as constants


def single_file_to_dict(in_path, demographics, context_list=None):
    demo_dict = {}
    df_all = pd.read_csv(in_path)
    for demo in demographics:
        df = df_all.loc[df_all["Text"].str.startswith(constants.VARIABLE_DICT[demo]), :]
        if context_list is None:
            demo_dict[demo] = df
        else:
            demo_dict[demo] = df.loc[
                df["Text"].apply(has_context, context_list=context_list), :
            ]
    return demo_dict


def has_context(txt, context_list):
    if isinstance(txt, str) and any(context in txt for context in context_list):
        return True
    else:
        return False


def mult_files_to_dict(in_path, demographics, context_list=None):
    demo_dict = {}
    for demo in demographics:
        for file in os.listdir(in_path):
            if file.endswith(".csv") and demo in file:
                if demo == "MALE_SINGLE" and not "FEMALE" in file:
                    df = pd.read_csv(os.path.join(in_path, file))
                elif demo != "MALE_SINGLE":
                    df = pd.read_csv(os.path.join(in_path, file))

                if context_list is None:
                    demo_dict[demo] = df
                else:
                    demo_dict[demo] = df.loc[
                        df["Text"].apply(has_context, context_list=context_list), :
                    ]
    return demo_dict


def add_english(demo, ethnic=False, abbreviated=True):
    if ethnic:
        tmp_dict = {
            "Der Türke": "/\nThe Turk (m)",
            "Die Türkin": "/\nThe Turk (f)",
            "Der Deutsche": "/\nThe Ger. (m)",
            "Die Deutsche": "/\nThe Ger. (f)",
        }
        abbrev_dict = {
            "Der Türke": "T (m)",
            "Die Türkin": "T (f)",
            "Der Deutsche": "G (m)",
            "Die Deutsche": "G (f)",
        }
    else:
        tmp_dict = {
            "Der Mann": "/\nThe man",
            "Die Frau": "/\nThe woman",
            "Er": "/He",
            "Sie": "/She",
        }
        abbrev_dict = {"Der Mann": "M", "Die Frau": "F"}
    if abbreviated:
        return abbrev_dict[demo]
    else:
        return demo + tmp_dict[demo]


def plot_regard_ratios(demo_dict, contexts, ax, ratios_df):
    print(ratios_df)
    colors = ["#B30524", "#F6AA8D", "#3B4CC0"]  # sns.color_palette("Spectral")
    dfs = []

    for demo, df in demo_dict.items():
        demo_name = constants.VARIABLE_DICT[demo]
        df["Demographic"] = add_english(demo_name, False, abbreviated=True)
        dfs.append(df)

    merged_df = pd.concat(dfs).reset_index()
    total = merged_df.groupby("Demographic")["Prediction"].count().reset_index()

    negative = (
        merged_df[merged_df.Prediction == 0]
        .groupby("Demographic")["Prediction"]
        .count()
        .reset_index()
    )
    neutral = (
        merged_df[merged_df.Prediction == 1]
        .groupby("Demographic")["Prediction"]
        .count()
        .reset_index()
    )
    positive = (
        merged_df[merged_df.Prediction == 2]
        .groupby("Demographic")["Prediction"]
        .count()
        .reset_index()
    )

    negative["Prediction"] = [
        i / j * 100 for i, j in zip(negative["Prediction"], total["Prediction"])
    ]

    neutral["Prediction"] = [
        i / j * 100 for i, j in zip(neutral["Prediction"], total["Prediction"])
    ]
    positive["Prediction"] = [
        i / j * 100 for i, j in zip(positive["Prediction"], total["Prediction"])
    ]

    bar1 = sns.barplot(
        x="Demographic",
        y="Prediction",
        data=negative,
        bottom=[i + j for i, j in zip(positive["Prediction"], neutral["Prediction"])],
        color=colors[0],
        ax=ax,
    )
    bar2 = sns.barplot(
        x="Demographic",
        y="Prediction",
        data=neutral,
        bottom=positive["Prediction"],
        color=colors[1],
        ax=ax,
    )
    bar3 = sns.barplot(
        x="Demographic", y="Prediction", data=positive, color=colors[-1], ax=ax
    )

    # add legend
    top_bar = mpatches.Patch(color=colors[0], label="negative")
    mid_bar = mpatches.Patch(color=colors[1], label="neutral")
    bottom_bar = mpatches.Patch(color=colors[-1], label="positive")

    if ax is not None:
        plt.legend(
            handles=[top_bar, mid_bar, bottom_bar],
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        ax.set_ylabel(contexts)
        ax.set_xlabel("")
        # plt.legend().remove()
    else:
        plt.legend(handles=[top_bar, mid_bar, bottom_bar])
