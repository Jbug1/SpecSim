# conatins funcitons for importing data
# this should include functions for reading in msps and cleaning/create sim datasets
import pandas as pd
import tools
import numpy as np
import scipy
import spectral_similarity
import copy


def get_adduct_subset(nist_df):

    return nist_df[
        (nist_df["precursor_type"] == "[M+H]+")
        | (nist_df["precursor_type"] == "[M-H]-")
    ]


def get_target_df(
    target_path,
    adduct_sub=False
):

    # get whole dataframe from msp files
    target_df = convert_msp_to_df_2(target_path)

    # get adduct subsets
    if adduct_sub:
        target_df = get_adduct_subset(target_df)

    # subset to only where we have real inchis
    target_df = target_df[target_df["inchi"] != ""]

    target_df["precursor"] = pd.to_numeric(target_df["precursor"])

    target_df.reset_index(inplace=True)

    return target_df


def add_gauss_noise_to_peaks(spec, scale_ratio):

    noises = np.zeros(len(spec))
    for i in range(len(spec)):
        noises[i] = np.random.normal(scale=spec[i][1] * scale_ratio)

    spec[:, 1] = spec[:, 1] + noises
    spec[:, 1] = np.clip(spec[:, 1], a_min=0, a_max=None)

    return spec


def add_noises_to_matches(matches, scale_ratio, mult):

    matches["query"] = matches.apply(
        lambda x: add_gauss_noise_to_peaks(x["query"], scale_ratio=scale_ratio), axis=1
    )
    matches["query"] = matches.apply(
        lambda x: add_beta_noise_to_spectrum(x["query"], x["query_prec"], mult=mult),
        axis=1,
    )

    return matches


def get_target_df_with_noise(
    target_path,
    noise_peaks=True,
    alter_existing_peaks=True,
    std=0,
    lam=0.0,
    num_peaks=None,
):

    # get whole dataframe from msp files
    target_df = convert_msp_to_df_2(target_path)

    # get adduct subsets
    target_df = get_adduct_subset(target_df)

    # subset to only where we have real inchis
    target_df = target_df[target_df["inchi"] != ""]

    target_df["precursor"] = pd.to_numeric(target_df["precursor"])

    target_df.reset_index(inplace=True)

    if alter_existing_peaks == True:
        target_df.apply(
            lambda x: add_gauss_noise_to_peaks(x["spectrum"], std=std), axis=1
        )

    if noise_peaks == True:
        target_df.apply(
            lambda x: add_beta_noise_to_spectrum(
                x["spectrum"], x["precursor"], num_peaks, lam
            ),
            axis=1,
        )

    return target_df


def get_spec_features(spec_query, precursor_query, spec_target, precursor_target, prec_remove=True):

    #if we have invalid spectra, return all -1s...shouldn't happen
    if len(spec_query) == 0 or len(spec_target) == 0:
        return list(np.zeros(8)-1)

    outrow = np.zeros(8)
    # first get all peaks below precursor mz

    below_prec_indices = np.where(
        spec_query[:, 0] < (precursor_query - tools.ppm(precursor_query, 3))
    )

    mass_reduction = np.sum(spec_query[below_prec_indices][:, 1]) / np.sum(
        spec_query[:, 1]
    )

    if prec_remove:
        spec_query = spec_query[below_prec_indices]

    n_peaks = len(spec_query)
    ent = scipy.stats.entropy(spec_query[:, 1])

    outrow[0] = ent
    outrow[1] = n_peaks

    if n_peaks < 2:
        outrow[2] = -1
    else:
        outrow[2] = ent / np.log(n_peaks)
    outrow[3] = mass_reduction

    below_prec_indices = np.where(
        spec_target[:, 0] < precursor_target - tools.ppm(precursor_target, 3)
    )
    mass_reduction = np.sum(spec_target[below_prec_indices][:, 1]) / np.sum(
        spec_target[:, 1]
    )

    spec_target = spec_target[below_prec_indices]
    n_peaks = len(spec_target)
    ent = scipy.stats.entropy(spec_target[:, 1])
    outrow[4] = ent
    outrow[5] = n_peaks
    if n_peaks < 2:
        outrow[6] = -1
    else:
        outrow[6] = ent / np.log(n_peaks)
    outrow[7] = mass_reduction

    return list(outrow)


def add_non_spec_features(query_row, target_row):
    """
    query prec
    target prec
    query ce
    target ce
    instrument same
    ce ratio
    ce abs
    prec abs
    prec ppm
    """

    outrow = np.zeros(9)

    # individual features
    outrow[0] = float(query_row["precursor"])
    outrow[1] = float(target_row["precursor"])
    outrow[2] = float(query_row["collision_energy"])
    outrow[3] = float(target_row["collision_energy"])

    # combined features
    outrow[4] = float(target_row["instrument"] == query_row["instrument"])

    if (
        float(target_row["collision_energy"]) > 0
        and float(query_row["collision_energy"]) > 0
    ):
        outrow[5] = max(
            float(target_row["collision_energy"]) / float(query_row["collision_energy"]),
            float(query_row["collision_energy"]) / float(target_row["collision_energy"]),
        )
    else:
        outrow[5] = 0

    outrow[6] = abs(
        float(target_row["collision_energy"]) - float(query_row["collision_energy"])
    )

    # precursor features
    outrow[7] = abs(query_row["precursor"] - target_row["precursor"])
    outrow[8] = abs(query_row["precursor"] - target_row["precursor"]) / tools.ppm(
        query_row["precursor"], 1
    )

    return outrow


def add_beta_noise_to_spectrum(spec, precursor_mz, mult, noise_peaks=None):

    if noise_peaks is None:
        noise_peaks = len(spec)

    # generate noise mzs and intensities to be added
    noise_spec = np.zeros((noise_peaks, 2))
    noise_spec[:, 1] = np.random.beta(a=1, b=5, size=noise_peaks) * mult
    noise_spec[:, 0] = np.random.uniform(0, precursor_mz, size=noise_peaks)

    # build the final spectrum with mzs and combined peaks
    spec = np.concatenate((spec, noise_spec))
    spec = spec[spec[:, 0].argsort()]
    return spec




def clean_and_spec_features(
    spec1,
    prec1,
    spec2,
    prec2,
    noise_thresh,
    centroid_thresh,
    centroid_type="ppm",
    power=1,
):
    """
    Function to clean the query and target specs according to parameters passed. Returns only matched spec
    """

    if centroid_type == "ppm":

        spec1_ = tools.clean_spectrum(
            spec1,
            noise_removal=noise_thresh,
            ms2_ppm=centroid_thresh,
            standardize=False,
            max_mz=prec1,
        )
        spec2_ = tools.clean_spectrum(
            spec2,
            noise_removal=noise_thresh,
            ms2_ppm=centroid_thresh,
            standardize=False,
            max_mz=prec2,
        )

    else:
        spec1_ = tools.clean_spectrum(
            spec1, noise_removal=noise_thresh, ms2_da=centroid_thresh, standardize=False
        )
        spec2_ = tools.clean_spectrum(
            spec2, noise_removal=noise_thresh, ms2_da=centroid_thresh, standardize=False
        )

    # reweight by given power
    spec1_ = tools.weight_intensity(spec1_, power)
    spec2_ = tools.weight_intensity(spec2_, power)

    # get new spec features
    spec_features = get_spec_features(spec1_, prec1, spec2_, prec2)

    # get spec change features
    # spec1_features = get_spec_change_features(spec1, spec1_)
    # spec2_features = get_spec_change_features(spec2, spec2_)

    spec1_ = tools.standardize_spectrum(spec1_)
    spec2_ = tools.standardize_spectrum(spec2_)

    return spec_features + [spec1_, spec2_] #spec1_features + spec2_features 

def get_sim_features(query, lib, methods, ms2_da=None, ms2_ppm=None):

    sims = spectral_similarity.multiple_similarity(
        query, lib, methods, ms2_da=ms2_da, ms2_ppm=ms2_ppm
    )
    return [sims[i] for i in methods]

def create_matches_df_new(query_df, target_df, precursor_thresh, max_rows_per_query, max_len, adduct_match):

    non_spec_columns = [
        "precquery",
        "prectarget",
        "cequery",
        "cetarget",
        "instsame",
        "ceratio",
        "ceabs",
        "prec_abs_dif",
        "prec_ppm_dif",
    ]
    seen = 0
    out = None
    target_df = target_df.sample(frac=1)
    printy = 1e5

    seen=0
    cores_set=set()
    for i in range(len(query_df)):
        
        seen+=1
        cores_set.add(query_df.iloc[i]['inchi_base'])

        if adduct_match:
            within_range = target_df[
                (abs(query_df.iloc[i]["precursor"] - target_df["precursor"])
                < tools.ppm(query_df.iloc[i]["precursor"], precursor_thresh)) & (query_df.iloc[i]["precursor_type"]==target_df["precursor_type"])
            ]

        else:
            within_range = target_df[
                (abs(query_df.iloc[i]["precursor"] - target_df["precursor"])
                < tools.ppm(query_df.iloc[i]["precursor"], precursor_thresh)) & (query_df.iloc[i]["mode"]==target_df["mode"])
            ]

        #catch case where there are no precursor matches
        if within_range.shape[0]==0:
            continue

        within_range = within_range.sample(frac=1)[:max_rows_per_query]

        within_range.reset_index(inplace=True)
        seen += len(within_range)

        if seen > printy:

            print(f"{seen} rows created")
            printy = printy + 1e5

        if out is None:
            out = within_range.apply(
                lambda x: add_non_spec_features(query_df.iloc[i], x),
                axis=1,
                result_type="expand",
            )
            out.columns = non_spec_columns

            out["query"] = [query_df.iloc[i]["spectrum"] for x in range(len(out))]
            out["target"] = within_range["spectrum"].tolist()
            out["match"] = (
                query_df.iloc[i]["inchi_base"] == within_range["inchi_base"]
            )

        else:
            temp = within_range.apply(
                lambda x: add_non_spec_features(query_df.iloc[i], x),
                axis=1,
                result_type="expand",
            )
            
            temp.columns = non_spec_columns
            

            temp["query"] = [query_df.iloc[i]["spectrum"] for x in range(len(temp))]
            temp["target"] = within_range["spectrum"]
            temp["match"] = (
                query_df.iloc[i]["inchi_base"] == within_range["inchi_base"]
            )
            out = pd.concat([out, temp])

        if len(out) >= max_len:
            return out

    print(f'total number of query spectra considered: {seen}')
    print(f'total inchicores seen: {len(cores_set)}')
    return out

def clean_and_spec_features_single(
    spec1,
    prec1,
    noise_thresh,
    centroid_thresh,
    centroid_type="ppm",
    power=1,
    verbose=False
):
    """
    Function to clean the query and target specs according to parameters passed. Returns only matched spec
    """

    if verbose:
        print(spec1)
    
    if centroid_type == "ppm":

        spec1_ = tools.clean_spectrum(
            spec1,
            noise_removal=noise_thresh,
            ms2_ppm=centroid_thresh,
            standardize=False,
            max_mz=prec1,
        )

    else:
        spec1_ = tools.clean_spectrum(
            spec1, noise_removal=noise_thresh, ms2_da=centroid_thresh, standardize=False
        )

    if verbose:
        print(spec1_)
        

    # reweight by given power
    spec1_ = tools.weight_intensity(spec1_, power)
    # print(spec1_)

    # get new spec features
    spec_features = get_spec_features_single(spec1_, prec1)
    if verbose:
        print(spec_features)
        print(yools)

    spec1_ = tools.standardize_spectrum(spec1_)

    return spec_features + [spec1_]


def get_spec_features_single(spec, precursor):

    if len(spec) == 0:
        spec = np.array([[1, 0]])

    outrow = np.zeros(4)

    # first get all peaks below precursor mz
    below_prec_indices = np.where(spec[:, 0] < (precursor - tools.ppm(precursor, 3)))
    mass_reduction = np.sum(spec[below_prec_indices][:, 1]) / np.sum(spec[:, 1])

    spec = spec[below_prec_indices]

    n_peaks = len(spec)
    ent = scipy.stats.entropy(spec[:, 1])

    outrow[0] = ent
    outrow[1] = n_peaks

    if n_peaks < 2:
        outrow[2] = -1
    else:
        outrow[2] = ent / np.log(n_peaks)
    outrow[3] = mass_reduction

    return list(outrow)


def get_sim_features_all(targets, queries, sim_methods, ms2_ppm=None, ms2_da=None):
    """
    This function calculates the similarities of the queries (one parameter setting) against all target specs
    """

    if ms2_da is None and ms2_ppm is None:
        raise ValueError("need either ms2da or ms2ppm to proceed")

    sims_out = None

    for i in range(targets.shape[1]):

        temp = pd.concat((targets.iloc[:, i : i + 1], queries), axis=1)

        col0 = temp.columns[0]
        col1 = temp.columns[1]

        sims = temp.apply(
            lambda x: get_sim_features(
                x[col0], x[col1], methods=sim_methods, ms2_da=ms2_da, ms2_ppm=ms2_ppm
            ), 
            axis=1,
            result_type="expand"
        )

        if sims_out is None:
            sims_out = sims
        else:
            sims_out = pd.concat((sims_out, sims), axis=1)

    return sims_out


def create_model_dataset(
    matches_df,
    sim_methods=None,
    noise_threshes=[0.01],
    centroid_tolerance_vals=[0.05],
    centroid_tolerance_types=["da"],
    powers=['orig'],
    prec_removes=[True]
):
    """ """
    # create helper vars
    out_df = None
    spec_columns = [
        "ent_query",
        "npeaks_query",
        "normalent_query",
        "mass_reduction_query",
        "ent_target",
        "npeaks_target",
        "normalent_target",
        "mass_reduction_target",
    ]

    # create initial value spec columns
    #
    for remove in prec_removes:
        
        init_spec_df = matches_df.apply(
            lambda x: get_spec_features(
                x["query"], x["precquery"], x["target"], x["prectarget"],remove
            ),
            axis=1,
            result_type="expand",
        )

        init_spec_df.columns = spec_columns

        ticker = 0
        for i in noise_threshes:
            for j in powers:
                for k in range(len(centroid_tolerance_vals)):

                    ticker += 1
                    if ticker % 10 == 0:
                        print(f"added {ticker} settings")

                    spec_columns_ = [
                        f"{x}_{i}_{j}_{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]}_{remove}"
                        for x in spec_columns
                    ]


                    sim_columns_ = [
                        f"{x}_{i}_{j}_{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]}_{remove}"
                        for x in sim_methods
                    ]

                    # clean specs and get corresponding spec features
                    cleaned_df = matches_df.apply(
                        lambda x: clean_and_spec_features(
                            x["query"],
                            x["precquery"],
                            x["target"],
                            x["prectarget"],
                            noise_thresh=i,
                            centroid_thresh=centroid_tolerance_vals[k],
                            power=j,
                        ),
                        axis=1,
                        result_type="expand",
                    )

                    cleaned_df.columns = (
                        spec_columns_  + ["query", "target"]
                    )

                    for x in range(len(cleaned_df)):

                        if (
                            np.isnan(cleaned_df.iloc[x]["query"]).any()
                            or np.isnan(cleaned_df.iloc[x]["target"]).any()
                        ):

                            print(
                                f"nans at row {x} under param setting {i}_{j}_{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]}"
                            )

                    # create columns of similarity scores
                    if centroid_tolerance_types[k] == "ppm":
                        sim_df = cleaned_df.apply(
                            lambda x: get_sim_features(
                                x["query"],
                                x["target"],
                                sim_methods,
                                ms2_ppm=centroid_tolerance_vals[k],
                            ),
                            axis=1,
                            result_type="expand",
                        )

                        sim_df.columns = sim_columns_

                    else:

                        sim_df = cleaned_df.apply(
                            lambda x: get_sim_features(
                                x["query"],
                                x["target"],
                                sim_methods,
                                ms2_da=centroid_tolerance_vals[k],
                            ),
                            axis=1,
                            result_type="expand",
                        )

                        sim_df.columns = sim_columns_

                    # add everything to the output df
                    if out_df is None:

                        out_df = pd.concat(
                            (
                                matches_df.iloc[:, :-3],
                                init_spec_df,
                                cleaned_df.iloc[:, :-2],
                                sim_df,
                            ),
                            axis=1,
                        )

                    else:

                        out_df = pd.concat(
                            (
                                out_df,
                                cleaned_df.iloc[:, :-2],
                                sim_df,
                            ),
                            axis=1,
                        )

    out_df["match"] = matches_df["match"]
    return out_df

def generate_keep_indices(noise_threshes, centroid_tolerance_vals, powers, spec_features, sim_methods, prec_keeps =[True],any_=False, nonspecs=False, init_spec=False):

    if nonspecs:
        keep_indices= list(range(9))
    else:
        keep_indices=list()

    if init_spec:
        keep_indices+=list(range(9,17))

    ind=17
    for _ in prec_keeps:
        for i in noise_threshes:
            for j in centroid_tolerance_vals:
                for k in powers:
                    
                    for l in spec_features:
                        if any_:
                            if True in [i,j,k,l,_]:
                                keep_indices.append(ind)
                        else:
                            if i==j==k==l==_==True:
                                keep_indices.append(ind)
                        ind+=1

                    for l in sim_methods:
                        
                        if any_:
                            if True in [i,j,k,l,_]:
                                keep_indices.append(ind)
                        else:
                            if i==j==k==l==_==True:
                                keep_indices.append(ind)
                        ind+=1

    return keep_indices

