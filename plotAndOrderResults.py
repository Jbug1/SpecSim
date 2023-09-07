import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import auc


def add_evals_to_df(dataframe):
    """
    add tpr, fpr, fdr, precision to df
    """
    for i in dataframe:
        if i != "metric":
            dataframe[i] = pd.to_numeric(dataframe[i])

    dataframe["tpr"] = dataframe["true_pos"] / (
        dataframe["true_pos"] + dataframe["false_neg"]
    )
    dataframe["fpr"] = dataframe["false_pos"] / (
        dataframe["true_neg"] + dataframe["false_pos"]
    )
    dataframe["tnr"] = 1 - dataframe["fpr"]
    dataframe["fdr"] = dataframe["false_pos"] / (
        dataframe["true_pos"] + dataframe["false_pos"]
    )
    dataframe["precision"] = 1 - dataframe["fdr"]
    dataframe["npv"] = dataframe["true_neg"] / (
        dataframe["true_neg"] + dataframe["false_neg"]
    )

    return dataframe


def order_criterion_from_df(dataframe, criterion="roc"):
    """
    order all metrics by AUC from a given dataframe
    """

    out_list_ordered = list()

    for i in list(set(dataframe["metric"])):

        subset = dataframe[dataframe["metric"] == i]

        if criterion.lower() == "roc":
            subset_met = auc(subset["fpr"], subset["tpr"])

        elif criterion.lower() == "prc":
            subset_met = auc(subset["tpr"][::-1], subset["precision"][::-1])

        elif criterion.lower() == "average_precision":

            recall = subset["tpr"].tolist()
            precision = subset["precision"].tolist()

            recall = recall[::-1][1:]
            precision = precision[::-1][1:]

            subset_met = 0.0
            total_recall = 0
            # subset_met=np.trapz(precision, recall)
            for j in range(1, len(precision)):

                subset_met += (recall[j] - recall[j - 1]) * (precision[j])
                total_recall += recall[j] - recall[j - 1]

            subset_met = subset_met + (1 - total_recall) * precision[0]

        elif criterion.lower() == "min_fdr":

            subset_met = min(subset["fdr"])

        elif criterion.lower() == "sample_prc":

            subset_met = 0
            for j in np.linspace(0.0, 0.9, 10):

                # get subset of subset where tpr is within this decile
                sub_sub = subset[(subset["tpr"] > j) & (subset["tpr"] < j + 0.1)]

                # sample 10 indices from this block of df, then take their average
                sample_inds = np.random.choice(
                    range(len(sub_sub)),
                    size=10,
                )
                subset_met += (
                    np.mean(sub_sub["precision"].to_numpy()[[sample_inds]]) / 10
                )

        elif criterion == "thresh_cross":

            recalls = subset["tpr"].tolist()
            subset_met = len(recalls)
            for j in range(len(recalls)):

                if np.isnan(recalls[j]) or recalls[j] < 0.05:

                    subset_met = j
                    break

        else:
            raise ValueError(
                "criterion is not roc or prc or average_precision or min_fdr or npv_ppv or sample_prc"
            )

        out_list_ordered.append((i, subset_met))

    out_list_ordered.sort(reverse=True, key=lambda x: 0 if np.isnan(x[1]) else x[1])

    return out_list_ordered


def plot_all_curves(dataframe, dataframe_name, curve_type="roc", highlights=[]):
    """
    function to recreate fig
    """

    # highlights = highlights + ['entropy', 'unweighted_entropy', 'cosine']
    print(highlights)

    if curve_type.lower() not in ["fdr", "roc", "prc", "npv_ppv"]:
        raise ValueError("criterion is not roc or prc or fdr or npv_ppv")

    metrics = list(set(dataframe["metric"]))

    if curve_type.lower() == "roc":

        plt.ylabel("TPR", fontsize=15)
        plt.xlabel("FPR", fontsize=15)

        for i in metrics:

            subset = dataframe[dataframe["metric"] == i]

            if i not in highlights:

                plt.plot(subset["fpr"], subset["tpr"], c="tab:gray", alpha=0.05)

            if i in highlights:

                plt.plot(subset["fpr"], subset["tpr"], label=i)

    elif curve_type.lower() == "prc":

        plt.ylabel("Precision")
        plt.xlabel("Recall")

        for i in metrics:

            subset = dataframe[dataframe["metric"] == i]

            if i not in highlights:

                plt.plot(subset["tpr"], subset["precision"], c="tab:gray", alpha=0.05)

            if i in highlights:

                plt.plot(subset["tpr"], subset["precision"], label=i)

    elif curve_type.lower() == "fdr":

        xs = np.linspace(0, 1, num=2000)
        plt.ylabel("FDR")
        plt.xlabel("Similarity Threshold")

        for i in metrics:

            subset = dataframe[dataframe["metric"] == i]

            if i not in highlights:

                plt.plot(xs, subset["fdr"], c="tab:gray", alpha=0.05)

            if i in highlights:

                plt.plot(xs, subset["fdr"], label=i)

    elif curve_type.lower() == "npv_ppv":

        xs = np.linspace(0, 1, num=1000)
        plt.ylabel("PPV", fontsize=15)
        plt.xlabel("NPV")

        for i in metrics:

            subset = dataframe[dataframe["metric"] == i]

            if i not in highlights:

                plt.plot(subset["npv"], subset["precision"], c="tab:gray", alpha=0.05)

            if i in highlights:

                plt.plot(subset["npv"], subset["precision"], label=i)

    # final plot specifications
    plt.title(f"{curve_type.upper()} Curve for {dataframe_name}", fontsize=16)
    plt.rcParams["figure.figsize"] = (12, 12)
    plt.legend(prop={"size": 16})
    plt.show()

    return


def read_all_quantiles(quantiles_folder, plot_name, plot_metrics):

    quantile_sheets = os.listdir(quantiles_folder)
    quantiles = len(quantile_sheets)
    metrics = set(pd.read_csv(f"{quantiles_folder}\\{quantile_sheets[0]}")["metric"])

    if plot_metrics is None:
        plot_metrics = metrics

    metric_dict = dict()
    for i in metrics:
        metric_dict[i] = list()

    for quantile_sheet in quantile_sheets:

        df = pd.read_csv(f"{quantiles_folder}\\{quantile_sheet}")
        add_evals_to_df(df)
        scores = order_criterion_from_df(df, "roc")

        for i in scores:

            metric_dict[i[0]].append(i[1])

    for key, value in metric_dict.items():

        if key in plot_metrics:
            plt.plot(list(range(quantiles)), value, ".", label=key, markersize=24)

    plt.legend(prop={"size": 14})
    plt.title(plot_name, size=20)
    plt.ylabel("AUROC Score", size=20)
    plt.xlabel("Quantile", size=20)
    plt.rcParams["figure.figsize"] = (20, 12)
    plt.show()
