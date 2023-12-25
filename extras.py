
import numpy as np
import pandas as pd
import tools


# putting extra functions here for now to clean things up
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

def chunk_create_all_to_all(
        outpath,
        matches, 
        sim_methods,
        noise_threshes_query=[0.01],
        centroid_tolerance_vals_query=[0.05],
        centroid_tolerance_types_query=["da"],
        powers_query=[1],
        noise_threshes_target=[0.01],
        centroid_tolerance_vals_target=[0.05],
        centroid_tolerance_types_target=["da"],
        powers_target=[1],
):
    
    inds = [int(i*1e4) for i in range(int(len(matches)/1e4)+1)]

    for i in range(len(inds)-1):

        matches_df = matches.iloc[inds[i]:inds[i+1]]

        out_df = create_model_dataset_all_to_all(
            matches_df,
            sim_methods,
            noise_threshes_query,
            centroid_tolerance_vals_query,
            centroid_tolerance_types_query,
            powers_query,
            noise_threshes_target,
            centroid_tolerance_vals_target,
            centroid_tolerance_types_target,
            powers_target
        )

        if i ==0:
            out_df.to_csv(outpath,index=False)
        else:
            out_df.to_csv(outpath, mode='a', header=False, index=False)

        if i%10==0:
            print(f'processed {inds[i+1]} rows')
    


def create_model_dataset_all_to_all(
    matches,
    sim_methods,
    noise_threshes_query=[0.01],
    centroid_tolerance_vals_query=[0.05],
    centroid_tolerance_types_query=["da"],
    powers_query=[1],
    noise_threshes_target=[0.01],
    centroid_tolerance_vals_target=[0.05],
    centroid_tolerance_types_target=["da"],
    powers_target=[1],
    verbose=False
):

    spec_columns_query = [
        "ent_query",
        "npeaks_query",
        "normalent_query",
        "mass_reduction_query"]
    
    spec_columns_target = [
        "ent_target",
        "npeaks_target",
        "normalent_target",
        "mass_reduction_target",
    ]

    #helper variables that will hold transformed target specs and sims
    targets_df = None
    sims_outer_df = None

    # create initial value spec columns
    out_df = matches.apply(
        lambda x: get_spec_features(
            x["query"], x["query_prec"], x["target"], x["target_prec"]
        ),
        axis=1,
        result_type="expand",
    )

    out_df.columns = spec_columns_query + spec_columns_target

    ticker=0
    for i in noise_threshes_target:
        for j in powers_target:
            for k in range(len(centroid_tolerance_vals_target)):

                
                if verbose:
                    ticker+=1
                    if ticker %5==0:
                        print(f'processed {ticker} settings for targets')
                #create the spec columns specific to this parameter setting
                spec_columns_ = [
                    f"{x}_{i}_{j}_{centroid_tolerance_vals_target[k]}{centroid_tolerance_types_target[k]}"
                    for x in spec_columns_target
                ]

                # clean the target columns for this setting
                cleaned_df = matches.apply(
                    lambda x: clean_and_spec_features_single(
                        x["target"],
                        x["target_prec"],
                        noise_thresh=i,
                        centroid_thresh=centroid_tolerance_vals_target[k],
                        centroid_type=centroid_tolerance_types_target[k],
                        power=j,
                    ),
                    axis=1,
                    result_type="expand",
                )

                # add the proper column names
                cleaned_df.columns = spec_columns_ + [
                    f"target_{i}_{j}_{centroid_tolerance_vals_target[k]}_{centroid_tolerance_types_target[k]}"
                ]

                # add the target spec columns to the initial df, leave off the transformed target
                out_df = pd.concat((out_df, cleaned_df.iloc[:, :-1]), axis=1)

                # begin targets df if this is the first, else concat
                if targets_df is None:
                    targets_df = cleaned_df.iloc[:, -1:].copy()
                else:
                    targets_df = pd.concat(
                        (targets_df, cleaned_df.iloc[:, -1:]), axis=1
                    )

                del cleaned_df

    #create variable to store column names
    sim_names=list()

    # now that we have all the spec metrics from the target, do query
    ticker = 0
    for i in noise_threshes_query:
        for j in powers_query:
            for k in range(len(centroid_tolerance_vals_query)):

                if verbose:
                    ticker += 1
                    if ticker % 5 == 0:
                        print(f"added {ticker} settings")

                spec_columns_ = [
                    f"{x}_{i}_{j}_{centroid_tolerance_vals_query[k]}{centroid_tolerance_types_query[k]}"
                    for x in spec_columns_query
                ]

                # clean specs and get corresponding spec features
                cleaned_df = matches.apply(
                    lambda x: clean_and_spec_features_single(
                        x["query"],
                        x["query_prec"],
                        noise_thresh=i,
                        centroid_thresh=centroid_tolerance_vals_query[k],
                        centroid_type=centroid_tolerance_types_query[k],
                        power=j,
                        verbose=False
                    ),
                    axis=1,
                    result_type="expand",
                )

                cleaned_df.columns = spec_columns_ + [
                    f"query_{i}_{j}_{centroid_tolerance_vals_query[k]}_{centroid_tolerance_types_query[k]}"
                ]

                # add the target spec columns to the initial df, leave off the transformed target
                out_df = pd.concat((out_df, cleaned_df.iloc[:, :-1]), axis=1)

                # no more need for spec feature columns
                cleaned_df = cleaned_df.iloc[:, -1:]

                #take corresponding centroiding type for similarity
                if centroid_tolerance_types_query[k] == "ppm":
                    sim_df = get_sim_features_all(
                        targets_df,
                        cleaned_df,
                        sim_methods,
                        ms2_ppm=centroid_tolerance_vals_query[k],
                    )

                else:
                    sim_df = get_sim_features_all(
                        targets_df,
                        cleaned_df,
                        sim_methods,
                        ms2_ppm=centroid_tolerance_vals_query[k],
                    )

                # add all the similarities to the sims outer df
                if sims_outer_df is None:
                    sims_outer_df = sim_df.copy()
                else:
                    sims_outer_df = pd.concat((sims_outer_df, sim_df), axis=1)

                del sim_df


                # create column names for what we have just produced
                temp = [
                    x
                    + f"query_{i}_{j}_{centroid_tolerance_vals_query[k]}_{centroid_tolerance_types_query[k]}"
                    for x in targets_df.columns
                ]
                
                for i_ in temp:
                    for j_ in sim_methods:
                        sim_names.append(i_ + j_)

    sims_outer_df.columns = sim_names
    out_df = pd.concat((matches.iloc[:,:16],out_df, sims_outer_df), axis=1)
    out_df["match"] = matches["match"]
    return out_df


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

    for key, val in outdict.items():
        print(f'{key}: {len(val)}')

    return pd.DataFrame(outdict)

def convert_msp_to_df_5(filepath):
    """
    Metlin
    """

    with open(filepath) as f:
        lines = f.readlines()

    names = list()
    precursortype = list()
    num_peaks = list()
    precursors = list()
    inchis = list()
    inchis_firsts = list()
    isv = list()
    collision_gas = list()
    collision_energy = list()
    instrument = list()
    specs = list()
    mslvls=list()

    spec = list()
    inchi = ""
    collision_gas_ = ""
    collision_energy_ = -1
    isv_ = -1
    instrument_ = ""
    name=""
    prectype=""
    inchi=""
    precmz=""
    n_peak=""
    mslvl=""


    tot_rows = 0
    for i in lines:

        splitty = i.split(":")
        if splitty[0] == "\n":

            if len(spec)>0:

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

                num_peaks.append(n_peak)
                n_peak=""

                names.append(name)
                name=""

                precursors.append(precmz)
                precmz=""

                precursortype.append(prectype)
                prectype=""

                mslvls.append(mslvl)
                mslvl=""

                tot_rows += 1
            
            continue

        if splitty[0] == "Name":
            name= splitty[1].strip()

        elif splitty[0] == "PrecursorMZ":
                precmz  = splitty[1].split(",")[0].strip()

        elif splitty[0] == "Num peaks":
            n_peak = splitty[1].strip()

        elif splitty[0].strip()=='Synon':

            if splitty[-1][:2] == "03":
                prectype= splitty[-1].split(':')[-1][2:].strip()

            if splitty[-1][:2] == "28":
                inchi= splitty[-1][2:].strip()

            if splitty[-1].split(':')[-1][:2] == "00":
                mslvl= splitty[-1][2:].strip()

            if splitty[-1][:2] == "05":
                collision_energy_= splitty[-1].replace('05','').strip()
            
            if splitty[-1][:2] == "06":
                instrument_= splitty[-1].split(':')[-1][2:].strip()


        elif tools.is_digit(splitty[0].split()[0]):
            spec.append([float(splitty[0].split()[0]), float(splitty[0].split()[1])])

    outdict = {
        "name": names,
        "precursor_type": precursortype,
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

    for key, val in outdict.items():
        print(f'{key}: {len(val)}')

    return pd.DataFrame(outdict)

def convert_msp_to_df_7(filepath):
    """
    nist23
    """

    with open(filepath) as f:
        lines = f.readlines()

    names = list()
    precursortype = list()
    num_peaks = list()
    precursors = list()
    inchis = list()
    inchis_firsts = list()
    isv = list()
    collision_gas = list()
    collision_energy = list()
    instrument = list()
    instrument_type=list()
    specs = list()
    mslvls=list()

    spec = list()
    inchi = np.nan
    collision_gas_ = ""
    collision_energy_ = -1
    isv_ = -1
    instrument_ = ""
    instrument_type_=''
    name=""
    prectype=np.nan
    inchi=""
    precmz=np.nan
    n_peak=""
    mslvl=""

    tot_rows = 0
    for i in lines:

        #splitty = i.split(":")
        if "ID=" in i:

            if len(spec)>0:

                specs.append(np.array(spec))
                n_peak=len(spec)
                spec = list()

                inchis.append(inchi)

                if type(inchi)==str:
                    inchis_firsts.append(inchi.split("-")[0])
                else:
                    inchis_firsts.append(np.nan)
                inchi = np.nan

                collision_gas.append(collision_gas_)
                collision_gas_ = ""

                collision_energy.append(collision_energy_)
                collision_energy_ = -1

                isv.append(isv_)
                isv_ = -1

                instrument.append(instrument_)
                instrument_ = ""

                instrument_type.append(instrument_type_)
                instrument_type_ = ""

                num_peaks.append(n_peak)
                n_peak=""

                names.append(name)
                name=""

                precursors.append(precmz)
                precmz=np.nan

                precursortype.append(prectype)
                prectype=np.nan

                mslvls.append(mslvl)
                mslvl=""

                tot_rows += 1
            
            continue
        else:
            label = i.split('=')

        if label[0]=='INSTRUMENT':
            instrument_ = label[1].strip()

        elif label[0]=='INSTRUMENT_TYPE':
           instrument_type_ = label[1].strip()

        elif label[0]=='COLLISION_ENERGY':
            try:
                collision_energy_ = int(label[-1].split()[-1][:-2])
            except:
                print(label[-1].split()[-1][:-2])


        elif label[0]=='PRECURSOR_TYPE':
           prectype = label[1].strip()

        elif label[0]=='PRECURSOR_MZ':
            try:
                precmz = float(label[1].strip())
            except:
                pass

        elif label[0]=='INCHIKEY':
           inchi = label[1].strip()

        elif tools.is_digit(i.split()[0]) and len(i.split())==2:
            spec.append([float(i.split()[0]), float(i.split()[1])])

    outdict = {
        "name": names,
        "precursor_type": precursortype,
        "n_peaks": num_peaks,
        "precursor": precursors,
        "inchi": inchis,
        "inchi_base": inchis_firsts,
        "instrument": instrument_type,
        "collision_energy": collision_energy,
        "spectrum": specs,
    }

    for key, val in outdict.items():
        print(f'{key}: {len(val)}')

    return pd.DataFrame(outdict)


def convert_msp_to_df_6(filepath):
    """
    nist23
    """

    with open(filepath) as f:
        lines = f.readlines()

    names = list()
    precursortype = list()
    num_peaks = list()
    precursors = list()
    inchis = list()
    inchis_firsts = list()
    isv = list()
    collision_gas = list()
    collision_energy = list()
    instrument = list()
    specs = list()
    mslvls=list()

    spec = list()
    inchi = np.nan
    collision_gas_ = ""
    collision_energy_ = -1
    isv_ = -1
    instrument_ = ""
    name=""
    prectype=np.nan
    inchi=""
    precmz=np.nan
    n_peak=""
    mslvl=""

    tot_rows = 0
    for i in lines:

        #splitty = i.split(":")
        if "ID=" in i:

            if len(spec)>0:

                specs.append(np.array(spec))
                n_peak=len(spec)
                spec = list()

                inchis.append(inchi)

                if type(inchi)==str:
                    inchis_firsts.append(inchi.split("-")[0])
                else:
                    inchis_firsts.append(np.nan)
                inchi = np.nan

                collision_gas.append(collision_gas_)
                collision_gas_ = ""

                collision_energy.append(collision_energy_)
                collision_energy_ = -1

                isv.append(isv_)
                isv_ = -1

                instrument.append(instrument_)
                instrument_ = ""

                num_peaks.append(n_peak)
                n_peak=""

                names.append(name)
                name=""

                precursors.append(precmz)
                precmz=np.nan

                precursortype.append(prectype)
                prectype=np.nan

                mslvls.append(mslvl)
                mslvl=""

                tot_rows += 1
            
            continue

        if '[' in i:
            prectype = i.strip()

        elif tools.is_digit(i) and len(i.split())==1:
                precmz  = float(i)

        elif not tools.is_digit(i) and len(i.split())==1:
            inchi = i.strip()

        elif tools.is_digit(i.split()[0]) and len(i.split())==2:
            spec.append([float(i.split()[0]), float(i.split()[1])])

    outdict = {
        "name": names,
        "precursor_type": precursortype,
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

    for key, val in outdict.items():
        print(f'{key}: {len(val)}')

    return pd.DataFrame(outdict)


def convert_msp_to_df_3(filepath):
    """
    GNPS
    """

    with open(filepath) as f:
        lines = f.readlines()

    names = list()
    precursortype = list()
    num_peaks = list()
    precursors = list()
    inchis = list()
    inchis_firsts = list()
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
    name=""
    prectype=""
    inchi=""
    precmz=""
    n_peak=""


    tot_rows = 0
    for i in lines:

        splitty = i.split(":")
        if splitty[0] == "\n":

            if len(spec)>0:

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

                num_peaks.append(n_peak)
                n_peak=""

                names.append(name)
                name=""

                precursors.append(precmz)
                precmz=""

                precursortype.append(prectype)
                prectype=""

                tot_rows += 1
            
            continue

        if splitty[0] == "NAME":
            name = splitty[1].strip()

        if splitty[0] == "PRECURSORTYPE":
            prectype= splitty[1].strip()

        if splitty[0] == "INCHIKEY":
            inchi = splitty[1].strip()

        elif splitty[0] == "PRECURSORMZ":

            precmz  = splitty[1].split(",")[0].strip()

        elif splitty[0] == "Num Peaks":
            n_peak = splitty[1].strip()

        elif splitty[0] == "COLLISIONGAS":
            collision_gas_ = splitty[1].strip()

        elif splitty[0] == "COLLISIONENERGY":
            collision_energy_ = splitty[1].strip()

            for i in collision_energy_:

                if not tools.is_digit(i):

                    collision_energy_ = -1
                    break

        elif splitty[0] == "ISV":
            isv_ = splitty[1].strip()

            for i in isv_:

                if not tools.is_digit(i):

                    isv_ = -1
                    break

        elif splitty[0] == "INSTRUMENTTYPE":
            instrument_ = splitty[1].strip()

        elif tools.is_digit(splitty[0].split()[0]):
            spec.append([float(splitty[0].split()[0]), float(splitty[0].split()[1])])

    outdict = {
        "name": names,
        "precursor_type": precursortype,
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

    for key, val in outdict.items():
        print(f'{key}: {len(val)}')

    return pd.DataFrame(outdict)

def convert_msp_to_df_4(filepath):
    """
    Mona
    """

    with open(filepath) as f:
        lines = f.readlines()

    names = list()
    precursortype = list()
    num_peaks = list()
    precursors = list()
    inchis = list()
    inchis_firsts = list()
    isv = list()
    collision_gas = list()
    collision_energy = list()
    instrument = list()
    specs = list()
    spec_types=list()

    spec = list()
    inchi = ""
    collision_gas_ = ""
    collision_energy_ = -1
    isv_ = -1
    instrument_ = ""
    name=""
    prectype=""
    inchi=""
    precmz=""
    n_peak=""

    tot_rows = 0
    for i in lines:

        splitty = i.split(":")
        if splitty[0] == "\n":

            if len(spec)>0:

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

                num_peaks.append(n_peak)
                n_peak=""

                names.append(name)
                name=""

                precursors.append(precmz)
                precmz=""

                precursortype.append(prectype)
                prectype=""

                spec_types.append(spec_type)
                spec_type=""

                tot_rows += 1
                continue
            
            else:
                continue
            

        if splitty[0] == "Name":
            name = splitty[1].strip()

        if splitty[0] == "Spectrum_type":
            spec_type = splitty[1].strip()

        if splitty[0] == "Precursor_type":
            prectype= splitty[1].strip()

        if splitty[0] == "InChIKey":
            inchi = splitty[1].strip()

        elif splitty[0] == "ExactMass":

            precmz  = splitty[1].split(",")[0].strip()

        elif splitty[0] == "Num Peaks":
            n_peak = splitty[1].strip()

        elif splitty[0] == "Collision_gas":
            collision_gas_ = splitty[1].strip()

        elif splitty[0] == "Collision_energy":
            collision_energy_ = splitty[1].strip()[:-1]

            # for i in collision_energy_:

            #     if not tools.is_digit(i):

            #         collision_energy_ = -1
            #         break

        elif splitty[0] == "ISV":
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
        "precursor_type": precursortype,
        "n_peaks": num_peaks,
        "precursor": precursors,
        "inchi": inchis,
        "inchi_base": inchis_firsts,
        "isv": isv,
        "instrument": instrument,
        "collision_gas": collision_gas,
        "collision_energy": collision_energy,
        "spec_types":spec_types,
        "spectrum": specs,
    }

    for key, val in outdict.items():
        print(f'{key}: {len(val)}')

    return pd.DataFrame(outdict)



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

def get_spec_change_features(dirty, clean):

    if len(clean) == 0:
        clean = np.array([[1, 0]])
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
