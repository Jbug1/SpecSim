import math_distance
import tools

import numpy as np
from typing import Union

methods_range = {
    "entropy": [0, np.log(4)],
    "absolute_value": [0, 2],
    "avg_l": [0, 1.5],
    "bhattacharya_1": [0, np.arccos(0) ** 2],
    "bhattacharya_2": [0, np.inf],
    "canberra": [0, np.inf],
    "clark": [0, np.inf],
    "divergence": [0, np.inf],
    "euclidean": [0, np.sqrt(2)],
    "hellinger": [0, np.inf],
    "improved_similarity": [0, np.inf],
    "lorentzian": [0, np.inf],
    "manhattan": [0, 2],
    "matusita": [0, np.sqrt(2)],
    "mean_character": [0, 2],
    "motyka": [-0.5, 0],
    "ms_for_id": [-np.inf, 0],
    "pearson_correlation": [-1, 1],
    "penrose_shape": [0, np.sqrt(2)],
    "penrose_size": [0, np.inf],
    "probabilistic_symmetric_chi_squared": [0, 1],
    "similarity_index": [0, np.inf],
    "squared_chord": [0, 2],
    "squared_euclidean": [0, 2],
    "symmetric_chi_squared": [0, 0.5 * np.sqrt(2)],
    "vicis_symmetric_chi_squared_3": [0, 2],
    "wave_hedges": [0, np.inf],
    "whittaker_index_of_association": [0, np.inf],
    "perc_peaks_in_common_distance": [0, 1],
    "rbf": [0, 1],
    "chi2": [0, 1],
    "cosine_kernel": [0, 1],
    "laplacian": [0, 1],
    "minkowski": [0, 1],
    "correlation": [0, 1],
    "jensenshannon": [0, 1],
    "sqeuclidean": [0, 1],
    "min_revdot": [0, 1],
    "rbf_mod": [0, 1],
    "gini": [0, 1],
    "l2": [0, 1],
    "common_mass": [0, 1],
    "entropy_mod": [0, 1],
    "cross_ent": [0, 1],
    "binary_cross_ent": [0, 1],
    "kl": [0, 1],
    "chebyshev": [0, 1],
    "fidelity": [0, 1],
    "harmonic_mean": [0, 1],
    "ruzicka": [0, 1],
    "roberts": [0, 1],
    "intersection": [0, 1],
}


def similarity(
    spectrum_query: Union[list, np.ndarray],
    spectrum_library: Union[list, np.ndarray],
    method: str,
    ms2_ppm: float = None,
    ms2_da: float = None,
    need_clean_spectra: bool = True,
    need_normalize_result: bool = True,
) -> float:
    """
    Calculate the similarity between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Similarity between two spectra
    """
    if need_normalize_result:
        return 1 - distance(
            spectrum_query=spectrum_query,
            spectrum_library=spectrum_library,
            method=method,
            need_clean_spectra=need_clean_spectra,
            need_normalize_result=need_normalize_result,
            ms2_ppm=ms2_ppm,
            ms2_da=ms2_da,
        )
    else:
        return 0 - distance(
            spectrum_query=spectrum_query,
            spectrum_library=spectrum_library,
            method=method,
            need_clean_spectra=need_clean_spectra,
            need_normalize_result=need_normalize_result,
            ms2_ppm=ms2_ppm,
            ms2_da=ms2_da,
        )


def all_similarity(
    spectrum_query: Union[list, np.ndarray],
    spectrum_library: Union[list, np.ndarray],
    ms2_ppm: float = None,
    ms2_da: float = None,
    need_clean_spectra: bool = True,
    need_normalize_result: bool = True,
) -> dict:
    """
    Calculate all the similarity between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: A dict contains all similarity.
    """

    all_similarity_score = all_distance(
        spectrum_query=spectrum_query,
        spectrum_library=spectrum_library,
        need_clean_spectra=need_clean_spectra,
        need_normalize_result=need_normalize_result,
        ms2_ppm=ms2_ppm,
        ms2_da=ms2_da,
    )
    for m in all_similarity_score:
        if need_normalize_result:
            all_similarity_score[m] = 1 - all_similarity_score[m]
        else:
            all_similarity_score[m] = 0 - all_similarity_score[m]
    return all_similarity_score


def distance(
    spectrum_query: Union[list, np.ndarray],
    spectrum_library: Union[list, np.ndarray],
    method: str,
    ms2_ppm: float = None,
    ms2_da: float = None,
    need_clean_spectra: bool = True,
    need_normalize_result: bool = True,
) -> float:
    """
    Calculate the distance between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra
    """

    if ms2_ppm is None and ms2_da is None:
        raise ValueError("MS2 tolerance needs to be defined!")

    spectrum_query = np.asarray(spectrum_query, dtype=np.float32)
    spectrum_library = np.asarray(spectrum_library, dtype=np.float32)
    if need_clean_spectra:
        spectrum_query = tools.clean_spectrum(
            spectrum_query, ms2_ppm=ms2_ppm, ms2_da=ms2_da
        )
        spectrum_library = tools.clean_spectrum(
            spectrum_library, ms2_ppm=ms2_ppm, ms2_da=ms2_da
        )

    # Calculate similarity
    if spectrum_query.shape[0] > 0 and spectrum_library.shape[0] > 0:

        if "reverse" in method:
            spec_matched = match_peaks_in_spectra(
                spec_a=spectrum_query,
                spec_b=spectrum_library,
                ms2_ppm=ms2_ppm,
                ms2_da=ms2_da,
            )

            dist = math_distance.reverse_distance(
                spec_matched[:, 1], spec_matched[:, 2], metric=method.split("_")[0]
            )

        elif "max" in method:
            spec_matched = tools.match_peaks_in_spectra(
                spec_a=spectrum_query,
                spec_b=spectrum_library,
                ms2_ppm=ms2_ppm,
                ms2_da=ms2_da,
            )

            dist = math_distance.max_distance(
                spec_matched[:, 1], spec_matched[:, 2], metric=method.split("_")[0]
            )
        else:
            function_name = method + "_distance"
            if hasattr(math_distance, function_name):
                f = getattr(math_distance, function_name)
                spec_matched = match_peaks_in_spectra(
                    spec_a=spectrum_query,
                    spec_b=spectrum_library,
                    ms2_ppm=ms2_ppm,
                    ms2_da=ms2_da,
                )
                dist = f(spec_matched[:, 1], spec_matched[:, 2])

            elif hasattr(ms_distance, function_name):
                f = getattr(ms_distance, function_name)
                dist = f(
                    spectrum_query, spectrum_library, ms2_ppm=ms2_ppm, ms2_da=ms2_da
                )
            else:
                raise RuntimeError("Method name: {} error!".format(method))

        # Normalize result
        if need_normalize_result:
            if method not in methods_range:
                dist_range = [0, 1]
            else:
                dist_range = methods_range[method]

            dist = normalize_distance(dist, dist_range)

        return dist

    else:
        if need_normalize_result:
            return 1
        else:
            return np.inf


def multiple_distance(
    spectrum_query: Union[list, np.ndarray],
    spectrum_library: Union[list, np.ndarray],
    methods: list = None,
    ms2_ppm: float = None,
    ms2_da: float = None,
    need_clean_spectra: bool = True,
    need_normalize_result: bool = True,
) -> dict:
    """
    Calculate multiple distance between two spectra, find common peaks.
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.

    :param spectrum_query: The query spectrum, need to be in numpy array format.
    :param spectrum_library: The library spectrum, need to be in numpy array format.
    :param methods: A list of method names.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra
    """
    if methods is None:
        methods = (
            [i for i in methods_range]
            + [f"{i}_reverse" for i in methods_range]
            + [f"{i}_max" for i in methods_range]
        )

    result = {}
    for m in methods:
        dist = distance(
            spectrum_query=spectrum_query,
            spectrum_library=spectrum_library,
            method=m,
            need_clean_spectra=need_clean_spectra,
            need_normalize_result=need_normalize_result,
            ms2_ppm=ms2_ppm,
            ms2_da=ms2_da,
        )
        result[m] = float(dist)
    return result


def normalize_distance(dist, dist_range):
    if dist_range[1] == np.inf:
        if dist_range[0] == 0:
            result = 1 - 1 / (1 + dist)
        elif dist_range[1] == 1:
            result = 1 - 1 / dist
        else:
            raise NotImplementedError()
    elif dist_range[0] == -np.inf:
        if dist_range[1] == 0:
            result = -1 / (-1 + dist)
        else:
            raise NotImplementedError()
    else:
        result = (dist - dist_range[0]) / (dist_range[1] - dist_range[0])

    if result < 0:
        result = 0.0
    elif result > 1:
        result = 1.0

    return result
