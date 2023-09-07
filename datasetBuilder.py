# conatins funcitons for importing data
# this should include functions for reading in msps and cleaning/create sim datasets
import pandas as pd
import tools
import numpy as np
import scipy
import spectral_similarity


def get_adduct_subset(nist_df):

    return nist_df[
        (nist_df["precursor_type"] == "[M+H]+")
        | (nist_df["precursor_type"] == "[M-H]-")
    ]


def get_target_df(
    target_path,
):

    # get whole dataframe from msp files
    target_df = convert_msp_to_df_2(target_path)

    # get adduct subsets
    target_df = get_adduct_subset(target_df)

    # subset to only where we have real inchis
    target_df = target_df[target_df["inchi"] != ""]

    target_df["precursor"] = pd.to_numeric(target_df["precursor"])

    target_df.reset_index(inplace=True)

    return target_df


def add_gauss_noise_to_peaks(spec, std):

    noises = np.random.normal(scale=std, size=len(spec))
    spec[:, 1] = spec[:, 1] + noises
    spec[:, 1] = np.clip(spec[:, 1], a_min=0, a_max=None)

    return spec


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
            lambda x: add_poisson_noise_to_spectrum(
                x["spectrum"], x["precursor"], num_peaks, lam
            ),
            axis=1,
        )

    return target_df


def clean_grid(target_df, noise_thresholds, centroid_values, power_values):
    """
    This function creates many 'clean' spectra based off of values passed

    """
    new_df = list()
    clean_df = list()

    # create multiple spectra for each
    for i in range(len(target_df)):

        spec = target_df.iloc[i]["spectrum"]
        max_mz = target_df.iloc[i]["precursor"] - tools.ppm(
            target_df.iloc[i]["precursor"], 3
        )
        outs = list()
        clean_features = list()

        for j in noise_thresholds:
            for k in centroid_values:
                for l in power_values:

                    # get new spectrum and info from cleaning
                    new_spec, clean_features_ = tools.clean_spectrum(
                        spec, noise_removal=j, ms2_da=k, max_mz=max_mz
                    )

                    # add clean features to existing array
                    clean_features.append(clean_features_)
                    new_spec = tools.weight_intensity(new_spec, power=l)
                    outs.append(new_spec)

        new_df.append(outs)
        clean_df.append(clean_features)

    return pd.concat((target_df, pd.DataFrame(clean_df), pd.DataFrame(new_df)), axis=1)


def convert_msp_to_df(filepath):

    with open(filepath) as f:
        lines = f.readlines()

    names = list()
    precursortype = list()
    forms = list()
    num_peaks = list()
    precursors = list()
    inchis = list()
    inchis_firsts = list()
    specs = list()
    mslevel = list()
    isv = list()
    collision_gas = list()
    collision_energy = list()
    instrument = list()

    spec = list()
    inchi = ""
    collision_gas_ = ""
    collision_energy_ = -1
    isv_ = -1
    instrument_ = ""

    tot_rows = 0
    for i in lines:

        splitty = i.split(":")
        if splitty[0] == "\n":

            specs.append(np.array(spec))
            spec = list()

            inchis.append(inchi)
            inchis_firsts.append(inchi.split("-")[0])
            inchi = ""

            collision_gas.append(collision_gas_)
            collision_gas_ = ""

            collision_energy.append(collision_energy_)
            collision_energy_ = -1

            isv.append(isv_)
            isv_ = -1

            instrument.append(instrument_)
            instrument_ = ""

            tot_rows += 1
            continue

        if splitty[0] == "Name":
            names.append(splitty[1].strip())

        elif splitty[0] == "Synon" and len(splitty) > 2:

            if splitty[2][:2] == "03":

                precursortype.append(splitty[2][2:].strip())

            if splitty[2][:2] == "28":
                inchi = splitty[2][2:].strip()

            if splitty[2][:2] == "00":
                mslevel.append(splitty[2][2:].strip())

            if splitty[2][:2] == "16":
                isv_ = splitty[2][2:].strip()

                for i in isv_:

                    if not tools.is_digit(i):

                        isv_ = -1
                        break

            if splitty[2][:2] == "12":
                collision_gas_ = splitty[2][2:].strip()

            if splitty[2][:2] == "05":
                collision_energy_ = splitty[2][2:].strip()

                for i in collision_energy_:

                    if not tools.is_digit(i):

                        collision_energy_ = -1
                        break

            if splitty[2][:2] == "06":
                instrument_ = splitty[2][2:].strip()

        elif splitty[0] == "PrecursorMZ":

            precursors.append(splitty[1].split(",")[0].strip())

        elif splitty[0] == "Num peaks":
            num_peaks.append(splitty[1].strip())

        elif splitty[0] == "Formula":
            forms.append(splitty[1].strip())

        elif tools.is_digit(splitty[0].split()[0]):
            spec.append([float(splitty[0].split()[0]), float(splitty[0].split()[1])])

    outdict = {
        "name": names,
        "mslevel": mslevel,
        "precursor_type": precursortype,
        "formula": forms,
        "n_peaks": num_peaks,
        "precursor": precursors,
        "inchi": inchis,
        "inchi_base": inchis_firsts,
        "spectrum": specs,
        "isv": isv,
        "collision_gas": collision_gas,
        "collision_energy": collision_energy,
        "instrument": instrument,
    }

    return pd.DataFrame(outdict)


def convert_msp_to_df_2(filepath):

    with open(filepath) as f:
        lines = f.readlines()

    names = list()
    precursortype = list()
    forms = list()
    num_peaks = list()
    precursors = list()
    inchis = list()
    inchis_firsts = list()
    mslevel = list()
    isv = list()
    collision_gas = list()
    collision_energy = list()
    instrument = list()
    specs = list()

    spec = list()
    inchi = ""
    collision_gas_ = ""
    collision_energy_ = -1
    isv_ = -1
    instrument_ = ""

    tot_rows = 0
    for i in lines:

        splitty = i.split(":")
        if splitty[0] == "\n":

            specs.append(np.array(spec))
            spec = list()

            inchis.append(inchi)
            inchis_firsts.append(inchi.split("-")[0])
            inchi = ""

            collision_gas.append(collision_gas_)
            collision_gas_ = ""

            collision_energy.append(collision_energy_)
            collision_energy_ = -1

            isv.append(isv_)
            isv_ = -1

            instrument.append(instrument_)
            instrument_ = ""

            tot_rows += 1
            continue

        if splitty[0] == "Name":
            names.append(splitty[1].strip())

        if splitty[0] == "Precursor_type":
            precursortype.append(splitty[1].strip())

        if splitty[0] == "InChIKey":
            inchi = splitty[1].strip()

        if splitty[0] == "Spectrum_type":
            mslevel.append(splitty[1].strip())

        elif splitty[0] == "PrecursorMZ":

            precursors.append(splitty[1].split(",")[0].strip())

        elif splitty[0] == "Num Peaks":
            num_peaks.append(splitty[1].strip())

        elif splitty[0] == "Formula":
            forms.append(splitty[1].strip())

        elif splitty[0] == "Collision_gas":
            collision_gas_ = splitty[1].strip()

        elif splitty[0] == "Collision_energy":
            collision_energy_ = splitty[1].strip()

            for i in collision_energy_:

                if not tools.is_digit(i):

                    collision_energy_ = -1
                    break

        elif splitty[0] == "In-source_voltage":
            isv_ = splitty[1].strip()

            for i in isv_:

                if not tools.is_digit(i):

                    isv_ = -1
                    break

        elif splitty[0] == "Instrument_type":
            instrument_ = splitty[1].strip()

        elif tools.is_digit(splitty[0].split()[0]):
            spec.append([float(splitty[0].split()[0]), float(splitty[0].split()[1])])

    outdict = {
        "name": names,
        "mslevel": mslevel,
        "precursor_type": precursortype,
        "formula": forms,
        "n_peaks": num_peaks,
        "precursor": precursors,
        "inchi": inchis,
        "inchi_base": inchis_firsts,
        "isv": isv,
        "instrument": instrument,
        "collision_gas": collision_gas,
        "collision_energy": collision_energy,
        "spectrum": specs,
    }

    return pd.DataFrame(outdict)


def row_builder(
    query_row,
    target_row,
    start_ind,
    methods,
    mass_tolerances,
    tolerance_types,
    nonspec_features,
    spec_features,
    sim_features,
    all_to_all=False,
):

    outrow = list()

    # get nonspec features
    if nonspec_features:
        outrow = outrow + add_non_spec_features(query_row, target_row)

    # get spec features
    if spec_features:
        outrow = outrow + add_spec_features(query_row, target_row)

    # convert similarities to list of all similarities,
    if sim_features:
        outrow = outrow + get_cleaned_similarities(
            query_row[start_ind:],
            target_row[start_ind:],
            methods,
            mass_tolerances,
            tolerance_types=tolerance_types,
            all_to_all=all_to_all,
        )

    return outrow


def get_cleaned_similarities(
    query_specs, target_specs, methods, mass_tolerances, tolerance_types, all_to_all
):
    """
    need to pass this function the spectra only
    """
    sims_out = list()

    if len(query_specs) != len(target_specs):
        raise ValueError("mistake query and target specs are not the same length")

    if not all_to_all:
        for i in range(len(query_specs)):
            for j in range(len(mass_tolerances)):

                if tolerance_types[j] == "da":
                    # only do comparison for the same level of cleanliness
                    sims = spectral_similarity.multiple_similarity(
                        query_specs[i],
                        target_specs[i],
                        methods=methods,
                        ms2_da=mass_tolerances[j],
                    )

                elif tolerance_types[j] == "ppm":
                    sims = spectral_similarity.multiple_similarity(
                        query_specs[i],
                        target_specs[i],
                        methods=methods,
                        ms2_ppm=mass_tolerances[j],
                    )
                else:
                    raise ValueError("Mass tolerance type is not either da or ppm")

                # unpack dictionary to array and append to big array
                sims_array = [sims[x] for x in methods]
                sims_out = sims_out + sims_array

    else:
        for i in range(len(query_specs)):
            for j in range(len(mass_tolerances)):
                for k in range(len(target_specs)):

                    if i < k:
                        continue

                    if tolerance_types[j] == "da":
                        # only do comparison for the same level of cleanliness
                        sims = spectral_similarity.multiple_similarity(
                            query_specs[i],
                            target_specs[k],
                            methods=methods,
                            ms2_da=mass_tolerances[j],
                        )

                    elif tolerance_types[j] == "ppm":
                        sims = spectral_similarity.multiple_similarity(
                            query_specs[i],
                            target_specs[k],
                            methods=methods,
                            ms2_ppm=mass_tolerances[j],
                        )
                    else:
                        raise ValueError("Mass tolerance type is not either da or ppm")

                    # unpack dictionary to array and append to big array
                    sims_array = [sims[x] for x in methods]
                    sims_out = sims_out + sims_array

    return list(sims_out)


def get_spec_change_features(dirty, clean):

    ent_change = scipy.stats.entropy(dirty[:, 1]) - scipy.stats.entropy(clean[:, 1])
    peaks_change = len(dirty) - len(clean)

    # avoid div0 error
    if len(clean) < 2:
        normal_ent_change = -1
    else:
        normal_ent_change = scipy.stats.entropy(dirty[:, 1]) / np.log(
            len(dirty)
        ) - scipy.stats.entropy(clean[:, 1]) / np.log(len(clean))
    mass_change = sum(clean[:, 1]) - sum(dirty[:, 1])

    return [ent_change, peaks_change, normal_ent_change, mass_change]


def get_spec_features(spec_query, precursor_query, spec_target, precursor_target):

    outrow = np.zeros(8)
    # first get all peaks below precursor mz

    below_prec_indices = np.where(
        spec_query[:, 0] < (precursor_query - tools.ppm(precursor_query, 3))
    )
    mass_reduction = np.sum(spec_query[below_prec_indices][:, 1]) / np.sum(
        spec_query[:, 1]
    )

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

    outrow = np.zeros(14)

    # individual features
    outrow[0] = int(query_row["isv"])
    outrow[1] = int(query_row["collision_energy"])
    outrow[2] = int(target_row["isv"])
    outrow[3] = int(target_row["collision_energy"])

    # combined features
    outrow[4] = int(target_row["collision_gas"] == query_row["collision_gas"])
    outrow[5] = int(target_row["instrument"] == query_row["instrument"])
    outrow[6] = int(
        int(target_row["collision_energy"]) == int(query_row["collision_energy"])
    )
    outrow[7] = int(int(target_row["isv"]) == int(query_row["isv"]))

    if int(target_row["isv"]) > 0 and int(query_row["isv"]) > 0:
        outrow[8] = max(
            int(target_row["isv"]) / int(query_row["isv"]),
            int(query_row["isv"]) / int(target_row["isv"]),
        )
    else:
        outrow[8] = 0

    if (
        int(target_row["collision_energy"]) > 0
        and int(query_row["collision_energy"]) > 0
    ):
        outrow[9] = max(
            int(target_row["collision_energy"]) / int(query_row["collision_energy"]),
            int(query_row["collision_energy"]) / int(target_row["collision_energy"]),
        )
    else:
        outrow[9] = 0

    outrow[10] = abs(int(target_row["isv"]) - int(query_row["isv"]))
    outrow[11] = abs(
        int(target_row["collision_energy"]) - int(query_row["collision_energy"])
    )

    # precursor features
    outrow[12] = abs(query_row["precursor"] - target_row["precursor"])
    outrow[13] = abs(query_row["precursor"] - target_row["precursor"]) / tools.ppm(
        query_row["precursor"], 1
    )

    return outrow


def add_poisson_noise_to_spectrum(spec, precursor_mz, noise_peaks, lam):

    if noise_peaks is None:
        noise_peaks = len(spec)

    # generate noise mzs and intensities to be added
    noise_spec = np.zeros((noise_peaks, 2))
    noise_spec[:, 1] = np.random.poisson(lam=lam, size=noise_peaks)
    noise_spec[:, 0] = np.random.uniform(0, precursor_mz, size=noise_peaks)

    # build the final spectrum with mzs and combined peaks
    spec = np.concatenate((spec, noise_spec))
    spec = spec[spec[:, 0].argsort()]

    return spec


def create_model_dataset_old(
    target_df,
    precursor_mass_thresh,
    sim_methods=None,
    limit_rows=None,
    mass_tolerances=[0.05],
    tolerance_types=["da"],
    nonspec_features=True,
    spec_features=True,
    sim_features=True,
    all_to_all=False,
    entry_limit=None,
):
    """
    create dataset on which to train all models for match prediction

    target_df: pd Dataframe after having had clea_grid run. all the raw specs and other info as input
    precursor_mass_thresh: num: compare specs with precursors in this range of each other (ppm)
    sim_methods: list: names of similarity methods on which to score spec match
    limit_rows: roughly limit the created dataset to this number of rows
    mass_tolerances: num: either da or ppm thresholds for peak matching
    tolerance_types: list: for each mass tolerance, either 'da' or 'ppm' to interpret the number
    nonspec_features: bool: whether to add non-spectral features (metadata) to dataset
    spec_features: bool: whether to add spectral features (not similarities) as columns to dataset
    sim_features: bool: whether to add similarities defined in sim methods to dataset
    all_to_all: bool: whether to calculate similarities of all specs to each other...vs just specs with same cleaning/weight scheme
    entry_limit: num: the maximum number of rows that one query spectrum can contribute to the dataset
    """
    non_spec_columns = [
        "isvt",
        "cet",
        "npt",
        "entt",
        "cgsame",
        "instsame",
        "cesame",
        "isvsame",
        "isvratio",
        "ceratio",
        "isvabs",
        "ceabs",
        "prec_abs_dif",
        "prec_ppm_dif",
    ]
    outrows = list()

    # grab column at which the cleaned spectra begin
    spec_start = target_df.columns.get_loc("spectrum") + 1

    for i in range(len(target_df)):

        # grab row and search for all rows within precursor window
        query_row = target_df.iloc[i]
        err = tools.ppm(query_row["precursor"], precursor_mass_thresh)

        precursor_window_df = target_df[
            (abs(target_df["precursor"] - query_row["precursor"]) < err)
            & (target_df["precursor_type"] == query_row["precursor_type"])
        ]

        # shuffle results
        precursor_window_df = precursor_window_df.sample(frac=1)

        # grab maximum of entry_limit number of rows to contribute to dataset
        for j in range(len(precursor_window_df[:entry_limit])):

            target_row = precursor_window_df.iloc[j]

            # create boolean match flag so we know if inchi key are the same
            match_flag = (
                query_row["inchi"].split("-")[0] == target_row["inchi"].split("-")[0]
            )

            # create row
            outrow = row_builder(
                query_row,
                target_row,
                start_ind=spec_start,
                methods=sim_methods,
                mass_tolerances=mass_tolerances,
                tolerance_types=tolerance_types,
                nonspec_features=nonspec_features,
                spec_features=spec_features,
                sim_features=sim_features,
            )

            # then append match flag so we know our Y value!
            outrow = np.append(outrow, match_flag)

            # add similarities to what will eventually become out dataframe
            outrows.append(outrow)

        if limit_rows is not None:
            if len(outrows) >= limit_rows:

                print(
                    f"Used {i+1} entries to create Training Set of {len(outrows)} rows"
                )
                del target_df
                return pd.DataFrame(outrows)

    print(f"Used {i} entries to create Training Set of {len(outrows)} rows")
    return pd.DataFrame(outrows)


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
    spec1_features = get_spec_change_features(spec1, spec1_)
    spec2_features = get_spec_change_features(spec2, spec2_)

    spec1_ = tools.standardize_spectrum(spec1_)
    spec2_ = tools.standardize_spectrum(spec2_)

    return spec_features + spec1_features + spec2_features + [spec1_, spec2_]


def get_sim_features(query, lib, methods, ms2_da=None, ms2_ppm=None):

    sims = spectral_similarity.multiple_similarity(
        query, lib, methods, ms2_da=ms2_da, ms2_ppm=ms2_ppm
    )
    return [sims[i] for i in methods]


def create_matches_df(target_df, precursor_thresh, max_rows_per_query, max_len):

    non_spec_columns = [
        "cequery",
        "isvquery",
        "cetarget",
        "isvtarget",
        "cgsame",
        "instsame",
        "cesame",
        "isvsame",
        "isvratio",
        "ceratio",
        "isvabs",
        "ceabs",
        "prec_abs_dif",
        "prec_ppm_dif",
    ]
    seen = 0
    out = None
    target_df = target_df.sample(frac=1)
    printy = 1e5

    for i in range(len(target_df)):

        within_range = target_df[
            abs(target_df.iloc[i]["precursor"] - target_df["precursor"])
            < tools.ppm(target_df.iloc[i]["precursor"], precursor_thresh)
        ]
        within_range = within_range.sample(frac=1)[:max_rows_per_query]

        within_range.reset_index(inplace=True)
        seen += len(within_range)

        if seen > printy:

            print(f"{seen} rows created")
            printy = printy + 1e5

        if out is None:
            out = within_range.apply(
                lambda x: add_non_spec_features(target_df.iloc[i], x),
                axis=1,
                result_type="expand",
            )
            out.columns = non_spec_columns

            out["query_prec"] = target_df.iloc[i]["precursor"]
            out["target_prec"] = within_range["precursor"].tolist()
            out["query"] = [target_df.iloc[i]["spectrum"] for x in range(len(out))]

            out["target"] = within_range["spectrum"].tolist()
            out["match"] = [
                target_df.iloc[i]["inchi_base"] == within_range.iloc[x]["inchi_base"]
                for x in range(len(within_range))
            ]

        else:
            temp = within_range.apply(
                lambda x: add_non_spec_features(target_df.iloc[i], x),
                axis=1,
                result_type="expand",
            )

            temp.columns = non_spec_columns

            temp["query"] = [target_df.iloc[i]["spectrum"] for x in range(len(temp))]
            temp["query_prec"] = target_df.iloc[i]["precursor"]
            temp["target"] = within_range["spectrum"]
            temp["target_prec"] = within_range["precursor"]
            temp["match"] = (
                target_df.iloc[i]["inchi_base"] == within_range["inchi_base"]
            )
            out = pd.concat([out, temp])

        if len(out) >= max_len:
            return out

    return out


def create_model_dataset(
    matches_df,
    sim_methods=None,
    noise_threshes=[0.01],
    centroid_tolerance_vals=[0.05],
    centroid_tolerance_types=["da"],
    powers=[1],
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
    spec_change_columns = [
        "query_ent_change",
        "query_peaks_change",
        "query_normal_ent_change",
        "query_mass_change",
        "target_ent_change",
        "target_peaks_change",
        "target_normal_ent_change",
        "target_mass_change",
    ]

    # create initial value spec columns
    init_spec_df = matches_df.apply(
        lambda x: get_spec_features(
            x["query"], x["query_prec"], x["target"], x["target_prec"]
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
                    f"{x}_{i}_{j}_{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]}"
                    for x in spec_columns
                ]

                spec_change_columns_ = [
                    f"{x}_{i}_{j}_{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]}"
                    for x in spec_change_columns
                ]
                sim_columns_ = [
                    f"{x}_{i}_{j}_{centroid_tolerance_vals[k]}{centroid_tolerance_types[k]}"
                    for x in sim_methods
                ]

                # clean specs and get corresponding spec features
                cleaned_df = matches_df.apply(
                    lambda x: clean_and_spec_features(
                        x["query"],
                        x["query_prec"],
                        x["target"],
                        x["target_prec"],
                        noise_thresh=i,
                        centroid_thresh=centroid_tolerance_vals[k],
                        power=j,
                    ),
                    axis=1,
                    result_type="expand",
                )

                cleaned_df.columns = (
                    spec_columns_ + spec_change_columns_ + ["query", "target"]
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
