import tools
import spectral_similarity
import numpy as np
import time
import pandas as pd


def run_all_comparisons(
    target_df,
    decoy_df,
    precursor_mass_thresh,
    results_dict,
    metrics=None,
    compare_to_target=True,
):

    start = time.time()
    for i in range(len(target_df)):

        if i % 10000 == 0:
            print(f"examined {i} rows")
            print(time.time() - start)

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
                    target_row["spectrum"],
                    query_row["spectrum"],
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
                    target_row["spectrum"],
                    query_row["spectrum"],
                    match_flag,
                    metrics,
                    results_dict,
                )


def update_results_dict(spec_query, spec_reference, match_flag, metrics, results_dict):

    # grab all similarities with same parameters from the paper
    if metrics is None:
        similarities = spectral_similarity.all_similarity(
            spec_query, spec_reference, ms2_da=0.05
        )

    else:
        similarities = spectral_similarity.multiple_similarity(
            spec_query, spec_reference, methods=metrics, ms2_da=0.05
        )

    for i in similarities:

        sim = similarities[i]

        if sim > 1 or sim < 0:
            print("out of bounds")
            print(i)
            print(sim)
            print(spec_query)
            print(spec_reference)

        #         print(f' {i}: {sim}')

        # verify number of buckets for ROC curve
        buckets = results_dict[i].shape[0] - 2
        try:
            below_thresh_index = int(sim / (1 / buckets)) + 1
        except:
            print(i)
            print(sim)
            print(spec_query)
            print(spec_reference)
        # print(below_thresh_index)

        # these should be a match any negatives are false, any positives are true
        if match_flag:

            results_dict[i][:below_thresh_index][:, 0] += 1
            results_dict[i][below_thresh_index:][:, 3] += 1

        else:

            results_dict[i][:below_thresh_index][:, 1] += 1
            results_dict[i][below_thresh_index:][:, 2] += 1


def dict_to_df(dictionary):

    # out_df = pd.DataFrame(columns=['metric','true_pos','false_pos','true_neg','false_neg'])
    final = np.zeros((1, 5))
    for i in dictionary:

        entry_len = dictionary[i].shape[0]
        first = [i] * entry_len
        first = np.array([[i] for i in first])

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
