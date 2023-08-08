# conatins funcitons for importing data
# this should include functions for reading in msps and cleaning/create sim datasets
import pandas as pd
from . import tools
import numpy as np
import scipy
from . import spectral_similarity


def get_adduct_subset(nist_df):

    return nist_df[
        (nist_df["precursor_type"] == "[M+H]+")
        | (nist_df["precursor_type"] == "[M-H]-")
    ]


def get_target_df(target_path, noise_thresholds, centroid_values):

    # get whole dataframe from msp files
    target_df = convert_msp_to_df_2(target_path)

    # get adduct subsets
    target_df = get_adduct_subset(target_df)

    # subset to only where we have real inchis
    target_df = target_df[target_df["inchi"] != ""]

    target_df["precursor"] = pd.to_numeric(target_df["precursor"])

    target_df.reset_index(inplace=True)

    target_df = clean_grid(target_df, noise_thresholds, centroid_values)

    return target_df


def clean_grid(target_df, noise_thresholds, centroid_values, power_values):
    """
    This function creates many 'clean' spectra based off of values passed

    """
    new_df = list()
    # create multiple spectra for each
    for i in range(len(target_df)):

        spec = target_df.iloc[i]["spectrum"]
        max_mz = target_df.iloc[i]["precursor"] - tools.ppm(
            target_df.iloc[i]["precursor"], 5
        )
        outs = list()

        for j in noise_thresholds:
            for k in centroid_values:
                for l in power_values:

                    new_spec = tools.clean_spectrum(
                        spec, noise_removal=j, ms2_da=k, max_mz=max_mz
                    )
                    new_spec = tools.weight_intensity(new_spec, power=l)
                    outs.append(new_spec)

        new_df.append(outs)

    return pd.concat((target_df, pd.DataFrame(new_df)), axis=1)


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
    all_to_all,
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
            tolerance_typest=tolerance_types,
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
                        need_clean_spectra=False,
                    )

                elif tolerance_types[j] == "ppm":
                    sims = spectral_similarity.multiple_similarity(
                        query_specs[i],
                        target_specs[i],
                        methods=methods,
                        ms2_ppm=mass_tolerances[j],
                        need_clean_spectra=False,
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
                            need_clean_spectra=False,
                        )

                    elif tolerance_types[j] == "ppm":
                        sims = spectral_similarity.multiple_similarity(
                            query_specs[i],
                            target_specs[k],
                            methods=methods,
                            ms2_ppm=mass_tolerances[j],
                            need_clean_spectra=False,
                        )
                    else:
                        raise ValueError("Mass tolerance type is not either da or ppm")

                    # unpack dictionary to array and append to big array
                    sims_array = [sims[x] for x in methods]
                    sims_out = sims_out + sims_array

    return list(sims_out)


def add_spec_features(query_row, target_row):

    outrow = list()
    n_peaks = int(query_row["n_peaks"])
    ent = scipy.stats.entropy(query_row["spectrum"][:, 1])
    outrow.append(ent)
    outrow.append(n_peaks)
    outrow.append(ent / np.log(n_peaks))

    n_peaks = int(target_row["n_peaks"])
    ent = scipy.stats.entropy(target_row["spectrum"][:, 1])
    outrow.append(ent)
    outrow.append(n_peaks)
    outrow.append(ent / np.log(n_peaks))

    return outrow


def add_non_spec_features(query_row, target_row):

    outrow = list()
    # num features for query row
    #     outrow.append(int(query_row['isv']))
    #     outrow.append(int(query_row['collision_energy']))
    #     outrow.append(int(query_row['n_peaks']))
    #     outrow.append(spectral_entropy.calculate_entropy(query_row['spectrum']))

    #     #num features for target row
    # outrow.append(int(target_row['isv']))
    # outrow.append(int(target_row['collision_energy']))
    # #outrow.append(int(target_row['n_peaks']))
    # outrow.append(spectral_entropy.calculate_entropy(target_row['spectrum']))

    # combined features
    outrow.append(int(target_row["collision_gas"] == query_row["collision_gas"]))
    outrow.append(int(target_row["instrument"] == query_row["instrument"]))
    outrow.append(int(target_row["collision_energy"] == query_row["collision_energy"]))
    outrow.append(int(target_row["isv"]) == query_row["isv"])

    if int(target_row["isv"]) > 0 and int(query_row["isv"]) > 0:
        outrow.append(
            max(
                int(target_row["isv"]) / int(query_row["isv"]),
                int(query_row["isv"]) / int(target_row["isv"]),
            )
        )
    else:
        outrow.append(0)

    if (
        int(target_row["collision_energy"]) > 0
        and int(query_row["collision_energy"]) > 0
    ):
        outrow.append(
            max(
                int(target_row["collision_energy"])
                / int(query_row["collision_energy"]),
                int(query_row["collision_energy"])
                / int(target_row["collision_energy"]),
            )
        )
    else:
        outrow.append(0)

    outrow.append(abs(int(target_row["isv"]) - int(query_row["isv"])))
    outrow.append(
        abs(int(target_row["collision_energy"]) - int(query_row["collision_energy"]))
    )

    return outrow


def create_model_dataset(
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
