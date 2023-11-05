import tools
import spectral_similarity
import numpy as np
import time
import pandas as pd
from sklearn.metrics import roc_auc_score as auc


def best_model_select(models, train, val, test):
    
    model_errs = np.zeros(len(models))
    train['match']=train['match'].astype(int)
    val['match']=val['match'].astype(int)
    test['match']=test['match'].astype(int)

    best_model = None
    best_error=np.inf

    i=0
    for model in models:

        clf = sklearn.base.clone(model)
        clf.fit(train.iloc[:,:-1], train.iloc[:,-1])
        val_error = zero_one_loss(clf.predict(val.iloc[:,:-1]),val.iloc[:,-1:].to_numpy())

        if val_error < best_error:
            best_model=copy.deepcopy(clf)
            best_error = val_error

        del(clf)
        model_errs[i]=val_error
        i+=1

    return str(best_model), zero_one_loss(best_model.predict(test.iloc[:,:-1]),test.iloc[:,-1:].to_numpy()), model_errs

def best_models_by_subset(cols, train_sizes, models, train, val, test):

    res_dict=dict()
    for key, value in cols.items():
        res_dict[key]=list()

        for size in train_sizes:
    
            res_dict[key].append(best_model_select(models, train.iloc[:size,value+[-1]], val.iloc[:,value+[-1]], test.iloc[:,value+[-1]]))

            print(f'finished {key} for {size}')

    return res_dict

def create_variable_comparisons(noise_threshes, centroid_threshes, centroid_types, powers, sim_methods, matches, outfolder):

    #matches = datasetBuilder.create_matches_df(target,i,max_matches,size)
    filepath = f'{outfolder}/matches_{i}_ppm.csv'

    for j in noise_threshes:
        for k in range(len(centroid_threshes)):
            for l in powers:

                cleaned = matches.apply(lambda x: datasetBuilder.clean_and_spec_features(x['query'],
                                                                                            x['query_prec'],
                                                                                            x['target'],
                                                                                            x['target_prec'],
                                                                                            noise_thresh=j,
                                                                                            centroid_thresh = centroid_threshes[k],
                                                                                            centroid_type=centroid_types[k],
                                                                                            power=l
                                                                                            ), 
                                                                                            axis=1,
                                                                                            result_type='expand')
                
                
                cleaned=pd.DataFrame(cleaned)
                cleaned = cleaned.iloc[:,-2:]
                cleaned.columns = ['query','library']
                                                                            
                aucs = tests.run_metrics_models_auc(sim_methods,[],cleaned, tol_thresh = centroid_threshes[k], tol_type=centroid_types[k])
                aucs = auc_to_df(aucs, list(matches.iloc[:,-1]))
                aucs['clean_specs'] =  f'{j}_{centroid_threshes[k]}_{centroid_types[k]}_{l}'
                aucs.to_csv(filepath, mode='a', header=False)


def run_metrics_models_auc(metrics, models, test, tol_thresh, tol_type):
    """
    will run al metric aucs on test data for all model settings for default paper settings
    unless otherwise hardcoded

    models should be a tuple of (name, indices for input, trained_model)
    """

    #create dict to hold all scores
    aucs = dict()

    #make sure we have valid metric subset to look at
    if metrics is not None: 
        for i in metrics:
            aucs[i[0]]=auc(test.iloc[:,-1:],test.iloc[:,i[1]])

    else:

        metrics=list()
        for i in spectral_similarity.methods_range:
            metrics.append(i)
            metrics.append('max_'+i)
            # metrics.append('min_'+i)
            # metrics.append('ave_'+i)

        if tol_type == 'da':
            sims = test.apply(lambda x: spectral_similarity.multiple_similarity(x['query'],x['library'],methods =metrics, ms2_da = tol_thresh), axis=1)
        else:
            sims = test.apply(lambda x: spectral_similarity.multiple_similarity(x['query'],x['library'],methods =metrics, ms2_ppm = tol_thresh), axis=1)
        
        try:
            #sims = [x[i] for x in sims]
            pass
            #aucs[i]=auc(test.iloc[:,-1:],sims)
        except:
            print(f'error on {i}')
            # print(sims)
            
            # for i in range(len(sims)):
            #     if np.isnan(sims[i]):
            #         print(sims[i])
            #         print(test.iloc[i]['query'])
            #         print(test.iloc[i]['library'])
                    
            # print('tayslkdf')

    for i in models:

        sims = i[2].predict_proba(test.iloc[:,i[1]]).squeeze()

        if i[2].classes_[1]==True:
            sims=sims[:,1]
        else:
            sims=sims[:,0]  

        aucs[i[0]] = auc(test.iloc[:,-1:],sims)

    return sims



def add_poisson_noise_to_spectrum(spec, precursor_mz, noise_peaks):

    # generate noise mzs and intensities to be added
    noise_intensities = np.random.poisson(lam=1, size=noise_peaks)
    noise_mzs = np.random.uniform(0, precursor_mz, size=noise_peaks)

    # sort noise_mzs
    noise_mzs.sort()

    # make noise spectrum
    noise_spec = list()
    for i in range(len(noise_mzs)):

        noise_spec.append([noise_mzs[i], noise_intensities[i]])

    noise_spec = np.array(noise_spec)

    # build the final spectrum with mzs and combined peaks
    out_spec = list()
    i = 0
    j = 0
    while i < len(spec) and j < len(len(noise_spec)):

        if spec[i][0] < noise_spec[j][0]:

            out_spec.append(spec[i])
            i += 1

        else:
            out_spec.append(noise_spec[j])
            j += 1

    while i < len(spec):

        out_spec.append(spec[i])
        i += 1

    while j < len(noise_spec):

        out_spec.append(noise_spec[j])
        j += 1

    return np.array(noise_spec)


def run_all_comparisons_models(test_dataset, precursor_mass_thresh, models):

    results_dict = dict()

    for i in range(len(models)):
        results_dict[i] = np.zeros((1000, 4))

    x = test_dataset[:, :-1]
    y = test_dataset[:, -1]

    for i in range(len(models)):
        y_hat = models[i].predict_proba(x)
        buckets = results_dict[i].shape[0] - 2

        for j in range(len(y_hat)):

            sim = y_hat[j][1]
            below_thresh_index = int(sim / (1 / buckets))

        if y[j] == True:

            results_dict[i][:below_thresh_index][:, 0] += 1
            results_dict[i][below_thresh_index:][:, 3] += 1

        else:

            results_dict[i][:below_thresh_index][:, 1] += 1
            results_dict[i][below_thresh_index:][:, 2] += 1


def run_all_comparisons(
    target_df,
    decoy_df=[],
    precursor_mass_thresh=10,
    metrics=None,
    compare_to_target=True,
    lim_rows=None,
):
    """
    This function compares all similarity metrics to each other, constructing a df of
    metrics used in evaluations for each one

    target_df: dataframe with actual spectra
    decoy_df: dataframe with decoy spectra
    precursor_mass_thresh: ppm num type
    results_dict:
    """
    results_dict = dict()

    if metrics == None:

        metrics = list()
        for i in spectral_similarity.methods_range:
            metrics.append(i)
            metrics.append("max_" + i)
            # metrics.append("reverse_" + i)
            metrics.append("min_" + i)
            metrics.append("ave_" + i)

    for i in metrics:
        results_dict[i] = np.zeros((1000, 4))

    start = time.time()
    for i in range(len(target_df)):

        # print(i)
        if i == lim_rows:
            return results_dict

        if i % 1000 == 0:
            print(f"examined {i} rows")
            # print(time.time() - start)

        query_row = target_df.iloc[i]

        err = tools.ppm(query_row["precursor"], precursor_mass_thresh)
        upper = query_row["precursor"] + err
        lower = query_row["precursor"] - err

        # grab all spectra in precursor range of same ion type
        precursor_window_df_target = target_df[
            (target_df["precursor"] > lower)
            & (target_df["precursor"] < upper)
            & (target_df["precursor_type"] == query_row["precursor_type"])
        ]

        if len(decoy_df) > 0:

            # grab all spectra in precursor range of same ion type
            precursor_window_df_decoy = decoy_df[
                (decoy_df["precursor"] > lower)
                & (decoy_df["precursor"] < upper)
                & (decoy_df["precursor_type"] == query_row["precursor_type"])
            ]
        # print(i)
        if compare_to_target:
            for j in range(len(precursor_window_df_target)):

                target_row = precursor_window_df_target.iloc[j]

                # if target_row['index']!=query_row['index']:

                # create boolean match flag so we know if inchi key are the same
                match_flag = (
                    query_row["inchi"].split("-")[0]
                    == target_row["inchi"].split("-")[0]
                )

                # call update function to update dictionary
                update_results_dict(
                    target_row[0],
                    query_row[0],
                    match_flag,
                    metrics,
                    results_dict,
                )

        if len(decoy_df) > 0:
            for j in range(len(precursor_window_df_decoy)):

                target_row = precursor_window_df_decoy.iloc[j]

                # create boolean match flag so we know if inchi key are the same
                match_flag = (
                    query_row["inchi"].split("-")[0]
                    == target_row["inchi"].split("-")[0]
                )

                # call update function to update dictionary
                a = update_results_dict(
                    target_row[0],
                    query_row[0],
                    match_flag,
                    metrics,
                    results_dict,
                )

    return results_dict


def update_results_dict(spec_query, spec_reference, match_flag, metrics, results_dict):
    # set metrics to None to compute all similarities
    # grab all similarities with same parameters from the paper

    similarities = spectral_similarity.multiple_similarity(
        spec_query, spec_reference, methods=metrics, ms2_da=0.05
    )

    for i in similarities:

        sim = similarities[i]

        # if sim > 1 or sim < 0:
        #     print("out of bounds")
        #     print(i)
        #     print(sim)
        #     print(spec_query)
        #     print(spec_reference)

        # verify number of buckets for ROC curve
        buckets = results_dict[i].shape[0] - 2
        try:
            below_thresh_index = int(sim / (1 / buckets)) + 1
        except:
            below_thresh_index = 0
            # print(i)
            # print(sim)
            # print(spec_query)
            # print(spec_reference)
        # print(below_thresh_index)

        # these should be a match any negatives are false, any positives are true
        if match_flag:

            results_dict[i][:below_thresh_index][:, 0] += 1
            results_dict[i][below_thresh_index:][:, 3] += 1

        else:

            results_dict[i][:below_thresh_index][:, 1] += 1
            results_dict[i][below_thresh_index:][:, 2] += 1


def dict_to_df(dictionary):
    """ """
    # create helper of right shape for concatentation of results
    final = np.zeros((1, 5))
    for i in dictionary:

        entry_len = dictionary[i].shape[0]
        first = [i] * entry_len
        first = np.array(first)
        # first = np.array([[i] for i in first])
        first = first.reshape(-1, 1)

        out = np.concatenate((first, dictionary[i]), axis=1)
        final = np.concatenate((final, out))

    return pd.DataFrame(
        final[1:], columns=["metric", "true_pos", "false_pos", "true_neg", "false_neg"]
    )


def run_comparisons_with_noise(
    target,
    decoy,
    precursor_mass_thresh,
    results_dict,
    decoy_breaks,
    base_filepath,
    metrics=[],
):

    decoy = decoy.sample(frac=1).reset_index(drop=True)

    # do the first pass on target data only
    run_all_comparisons(
        target,
        [],
        precursor_mass_thresh,
        results_dict,
        compare_to_target=True,
        metrics=metrics,
    )

    out = dict_to_df(results_dict)
    out.to_csv(f"{base_filepath}_target.csv")

    print("*" * 20)
    print("Ran on target DF")
    print("*" * 20)

    # how big of a block of decoy df do we need to grab?
    inc = int(len(decoy) / decoy_breaks)
    for i in range(decoy_breaks):

        # run the comparisons only picking up on the noise dataframe
        run_all_comparisons(
            target,
            decoy[i * inc : (i + 1) * inc],
            precursor_mass_thresh,
            results_dict,
            compare_to_target=False,
        )

        out = dict_to_df(results_dict)
        out.to_csv(f"{base_filepath}_{i}.csv")

        print("*" * 20)
        print(f"Ran on decoy {i}")
        print("*" * 20)
