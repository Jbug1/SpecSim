import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import auc
import datasetBuilder
import tools
import tests

########Figures Here######

def fig3(input, metrics, quant_variables, quantile_num, output_path):

    #create subdirectories by quant variable that has quantile pkls
    tests.break_data_into_quantiles(input, quant_variables, quantile_num, output_path)

    for i in quant_variables:

        aucs = tests.aucs_by_quantile(f'{output_path}/{i}', metrics)

        plot3_sub(aucs, i, f'{output_path}/{i}')

def plot3_sub(quantile_aucs, title, outpath):

    #reorganize the info such that we have a dict
    #with key: metric and x and y scatter
    final_dict = dict()
    for key, val in quantile_aucs.items():

        for i in range(len(val)):
            
            if val[i][0].split('_')[0] in final_dict:
                final_dict[val[i][0].split('_')[0]][0].append(key)
                final_dict[val[i][0].split('_')[0]][1].append(val[i][1])
            else:
                final_dict[val[i][0].split('_')[0]]= ([key], [val[i][1]])

    for key, val in final_dict.items():

        plt.scatter(val[0], val[1], label=key)
    
    plt.title(title)
    plt.xlabel('Quantile')
    plt.ylabel('AUC')
    plt.legend()

    plt.show()
    plt.savefig(f'{outpath}/figure.png')

def fig4(res_dict, indices, test):

    xs =list()
    ys=list()
    labels=list()

    for key, val in indices.keys():
        
        preds = res_dict[key][0].predict_proba(test.iloc[:,val])
        xs_, ys_ = tests.roc_curves_models(preds, test.iloc[:,-1].to_numpy)
        xs.append(xs_)
        ys.apend(ys_)
        labels.append(f'{key}: {res_dict[key][1]}')

    for i in range(len(xs)):

        plt.plot(xs[i],ys[i],label=labels[i])

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves by Feature Subset')
    plt.legend()
    plt.show()


def fig1(dir, matches_dir):
    """
    This directory will have n csvs that state overall metric performance on full sample

    This function will print the top 10 metrics for each and graph roc curves for best metric, entropy, dot, revdot
    """

    for i in os.listdir(dir):

        window = i.split('_')[0]
        res=pd.read_csv(f'{dir}/{i}', header=None)
        res.sort_values(by=2, inplace=True, ascending=False)

        print(f'Top Metrics for {i} by AUC')
        print(res.iloc[:10])
        print('\n')

        #get the parameters for top scoring metric
        name=res.iloc[0][1]
        noise, cent, cent_type, power = res.iloc[0][3].split('_')
        noise=float(noise)
        cent=float(cent)

        try:
            power=float(power)
        except:
            pass

        matches = pd.read_pickle(f'{matches_dir}/matches_{window}_ppm.pkl')

        metrics = ['cosine','entropy','reverse_dot_product']+[name]

        #get the curves for the 'traditional metrics'
        orig_res = datasetBuilder.create_model_dataset(matches,
                                                    sim_methods = ['cosine','entropy','reverse_dot_product'],
                                                    )
        #only maintain similarity columns and match
        orig_res = orig_res.iloc[:,16:]

        #get the scores for the top scoring result
        top_res =  datasetBuilder.create_model_dataset(matches,
                                                    sim_methods = [name],
                                                    noise_threshes=[noise],
                                                    centroid_tolerance_vals=[cent],
                                                    centroid_tolerance_types=[cent_type],
                                                    powers=[power]
                                                    )   
        #we only want similarity scores/match column
        top_res=top_res.iloc[:,-2:]
        
        orig_res=pd.concat((orig_res.iloc[:,:-1],top_res), axis=1)

        #execute function to get plot data
        names, xs, ys = tests.roc_curves_select_metrics(orig_res)

        for i in range(len(names)):
            orig_res.sort_values(by=names[i], inplace=True)
            auc_score = auc(orig_res[names[i]].to_numpy(),orig_res['match'].to_numpy())
            plt.plot(xs[i], ys[i], label=f'{metrics[i]}: {round(auc_score,2)}')

        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC Curves for Selected Metrics: {i}')
        plt.legend()
        plt.show()



def fig2(dir, ppm_windows):
    """Get top metric for each ppm window seeting by file"""

    for i in ppm_windows:

        #set collector vars for metrics
        res=dict()
        positions = dict()

        #set collector vars for vectorization settings
        res_vec=dict()
        positions_vec=dict()

        #keep track of total number of directories tested
        tots=0
        for j in os.listdir(dir):

            #catch weird other folders
            if tools.is_digit(j):
                tots+=1

                #read in metric scores, sort by AUROC
                mets = pd.read_csv(f'{dir}/{j}/{i}_ppm.csv', header=None)
                mets.sort_values(by=2, ascending=False, inplace=True)
                print(mets.head())
                #grab the first value of each metric, disregarding cleaning params
                mets_=mets.iloc[np.unique(mets[1], return_index=True)[1]]
                vecs= mets.iloc[np.unique(mets[3], return_index=True)[1]]
             
                #sort to order by auroc again
                mets_.sort_values(by=2, ascending=False, inplace=True)
                vecs.sort_values(by=2, ascending=False, inplace=True)
                
                #note the top metric
                if mets_.iloc[0,1] in res:
                    res[mets_.iloc[0,1]]+=1
                else:
                    res[mets_.iloc[0,1]]=1

                #note the rank of each metric
                for _ in range(len(mets_)):

                    if mets_.iloc[_,1] in positions:
                        positions[mets_.iloc[_,1]].append(_+1)
                    else:
                        positions[mets_.iloc[_,1]]=[_+1]


                #note the top vectorization setting
                if vecs.iloc[0,3] in res_vec:
                    res_vec[vecs.iloc[0,3]]+=1
                else:
                    res_vec[vecs.iloc[0,3]]=1

                #note the rank of each vectorization setting
                for _ in range(len(vecs)):

                    if vecs.iloc[_,3] in positions_vec:
                        positions_vec[vecs.iloc[_,3]].append(_+1)
                    else:
                        positions_vec[vecs.iloc[_,3]]=[_+1]

        #transform to % of time this metric is top
        for key, val in res.items():

            res[key]=val/tots

        #transform to mean ranking for this metric
        for key, val in positions.items():

            positions[key]=np.mean(val)

        #transform to % of time this vec_setting is top
        for key, val in res_vec.items():

            res_vec[key]=val/tots

        #transform to mean ranking for this vec setting
        for key, val in positions_vec.items():

            positions_vec[key]=np.mean(val)

        ranks = pd.DataFrame([res]).transpose().reset_index()
        ranks.sort_values(by=0, inplace=True, ascending=False)
        #top=np.array(ranks.iloc[:20]['index'].tolist())

        means = pd.DataFrame([positions]).transpose().reset_index()
        #means=means[np.isin(means['index'],top)]
        means.sort_values(by=0, inplace=True)

        ranks_vec = pd.DataFrame([res_vec]).transpose().reset_index()
        ranks_vec.sort_values(by=0, inplace=True, ascending=False)
        #top=np.array(ranks_vec.iloc[:20]['index'].tolist())

        means_vec = pd.DataFrame([positions_vec]).transpose().reset_index()
        #means=means[np.isin(means['index'],top)]
        means_vec.sort_values(by=0, inplace=True)

        print(f'Top Ranks and Means for {i} PPM: Metrics, {len(means)} total')
        print('Proportion of Time This Metric is Top')
        print(ranks.head(20))
        print('\n')

        print('Mean Ranking By Metric')
        print(means.head(20))
        print('\n')

        print(f'Top Ranks and Means for {i} PPM: Vectorization Settings: {len(means_vec)} total')
        print('Proportion of Time This Vectorization Setting is Top')
        print(ranks_vec.head(20))
        print('\n')

        print('Mean Ranking By Vectorization Setting')
        print(means_vec.head(20))
        print('\n')


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
