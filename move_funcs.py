# funcs to move over to new repo
import numpy as np
import scipy
from sklearn.metrics.pairwise import pairwise_kernels as pk
import scipy.spatial.distance as dist
import sklearn.metrics as met


def _select_common_peaks(p, q):
    select = q > 0
    p = p[select]
    p_sum = np.sum(p)
    if p_sum > 0:
        p = p / p_sum
    q = q[select]
    q = q / np.sum(q)
    return p, q


def _weight_intensity_by_entropy(
    x,
    ENTROPY_CUTOFF=3,
    WEIGHT_START=0.25,
    weight_slope=None,
):

    if weight_slope is None:
        weight_slope = (1 - WEIGHT_START) / ENTROPY_CUTOFF

    if np.sum(x) > 0:
        entropy_x = scipy.stats.entropy(x)

    if entropy_x < ENTROPY_CUTOFF:
        weight = WEIGHT_START + weight_slope * entropy_x
        x = np.power(x, weight)
        x_sum = np.sum(x)
        x = x / x_sum

    return x


def weight_intensity_by_entropy(x):
    """
    Jonah version of weight_intensity function
    """
    x = np.power(x, scipy.stats.entropy(x))
    return x / np.sum(x)


def weight_intensity(x, power=1):
    """
    Jonah version of weight_intensity function
    """

    x = np.power(x, power)
    return x / np.sum(x)


def unweighted_entropy_distance(p, q):
    r"""
    Unweighted entropy distance:

    .. math:

        -\frac{2\times S_{PQ}-S_P-S_Q} {ln(4)}, S_I=\sum_{i} {I_i ln(I_i)}
    """
    merged = p + q
    entropy_increase = (
        2 * scipy.stats.entropy(merged)
        - scipy.stats.entropy(p)
        - scipy.stats.entropy(q)
    )
    return entropy_increase


def euclidean_distance(p, q):
    r"""
    Euclidean distance:

    .. math::

        (\sum|P_{i}-Q_{i}|^2)^{1/2}
    """
    return np.sqrt(np.sum(np.power(p - q, 2)))


def manhattan_distance(p, q):
    r"""
    Manhattan distance:

    .. math::

        \sum|P_{i}-Q_{i}|
    """
    return np.sum(np.abs(p - q))


def chebyshev_distance(p, q):
    r"""
    Chebyshev distance:

    .. math::

        \underset{i}{\max}{(|P_{i}\ -\ Q_{i}|)}
    """
    return np.max(np.abs(p - q))


def squared_euclidean_distance(p, q):
    r"""
    Squared Euclidean distance:

    .. math::

        \sum(P_{i}-Q_{i})^2
    """
    return np.sum(np.power(p - q, 2))


def fidelity_distance(p, q):
    r"""
    Fidelity distance:

    .. math::

        1-\sum\sqrt{P_{i}Q_{i}}
    """
    return 1 - np.sum(np.sqrt(p * q))


def matusita_distance(p, q):
    r"""
    Matusita distance:

    .. math::

        \sqrt{\sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2}
    """
    return np.sqrt(np.sum(np.power(np.sqrt(p) - np.sqrt(q), 2)))


def squared_chord_distance(p, q):
    r"""
    Squared-chord distance:

    .. math::

        \sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2
    """
    return np.sum(np.power(np.sqrt(p) - np.sqrt(q), 2))


def bhattacharya_1_distance(p, q):
    r"""
    Bhattacharya 1 distance:

    .. math::

        (\arccos{(\sum\sqrt{P_{i}Q_{i}})})^2
    """
    s = np.sum(np.sqrt(p * q))
    if s > 1:
        s = 1
    return np.power(np.arccos(s), 2)


def bhattacharya_2_distance(p, q):
    r"""
    Bhattacharya 2 distance:

    .. math::

        -\ln{(\sum\sqrt{P_{i}Q_{i}})}
    """
    s = np.sum(np.sqrt(p * q))
    if s == 0:
        return np.inf
    else:
        return -np.log(s)


def harmonic_mean_distance(p, q):
    r"""
    Harmonic mean distance:

    .. math::

        1-2\sum(\frac{P_{i}Q_{i}}{P_{i}+Q_{i}})
    """
    return 1 - 2 * np.sum(p * q / (p + q))


def probabilistic_symmetric_chi_squared_distance(p, q):
    r"""
    Probabilistic symmetric χ2 distance:

    .. math::

        \frac{1}{2} \times \sum\frac{(P_{i}-Q_{i}\ )^2}{P_{i}+Q_{i}\ }
    """
    return 1 / 2 * np.sum(np.power(p - q, 2) / (p + q))


def ruzicka_distance(p, q):
    r"""
    Ruzicka distance:

    .. math::

        \frac{\sum{|P_{i}-Q_{i}|}}{\sum{\max(P_{i},Q_{i})}}
    """
    dist = np.sum(np.abs(p - q)) / np.sum(np.maximum(p, q))
    return dist


def roberts_distance(p, q):
    r"""
    Roberts distance:

    .. math::

        1-\sum\frac{(P_{i}+Q_{i})\frac{\min{(P_{i},Q_{i})}}{\max{(P_{i},Q_{i})}}}{\sum(P_{i}+Q_{i})}
    """
    return 1 - np.sum((p + q) / np.sum(p + q) * np.minimum(p, q) / np.maximum(p, q))


def intersection_distance(p, q):
    r"""
    Intersection distance:

    .. math::

        1-\frac{\sum\min{(P_{i},Q_{i})}}{\min(\sum{P_{i},\sum{Q_{i})}}}
    """
    return 1 - np.sum(np.minimum(p, q)) / min(np.sum(p), np.sum(q))


def motyka_distance(p, q):
    r"""
    Motyka distance:

    .. math::

        -\frac{\sum\min{(P_{i},Q_{i})}}{\sum(P_{i}+Q_{i})}
    """
    dist = np.sum(np.minimum(p, q)) / np.sum(p + q)
    return -dist


def canberra_distance(p, q):
    r"""
    Canberra distance:

    .. math::

        \sum\frac{|P_{i}-Q_{i}|}{|P_{i}|+|Q_{i}|}
    """
    return np.sum(np.abs(p - q) / (np.abs(p) + np.abs(q)))


def baroni_urbani_buser_distance(p, q):
    r"""
    Baroni-Urbani-Buser distance:

    .. math::

        1-\frac{\sum\min{(P_i,Q_i)}+\sqrt{\sum\min{(P_i,Q_i)}\sum(\max{(P)}-\max{(P_i,Q_i)})}}{\sum{\max{(P_i,Q_i)}+\sqrt{\sum{\min{(P_i,Q_i)}\sum(\max{(P)}-\max{(P_i,Q_i)})}}}}
    """
    if np.max(p) < np.max(q):
        p, q = q, p
    d1 = np.sqrt(np.sum(np.minimum(p, q) * np.sum(max(p) - np.maximum(p, q))))
    return 1 - (np.sum(np.minimum(p, q)) + d1) / (np.sum(np.maximum(p, q)) + d1)


def penrose_size_distance(p, q):
    r"""
    Penrose size distance:

    .. math::

        \sqrt N\sum{|P_i-Q_i|}
    """
    n = np.sum(p > 0)
    return np.sqrt(n) * np.sum(np.abs(p - q))


def mean_character_distance(p, q):
    r"""
    Mean character distance:

    .. math::

        \frac{1}{N}\sum{|P_i-Q_i|}
    """
    n = np.sum(p > 0)
    return 1 / n * np.sum(np.abs(p - q))


def lorentzian_distance(p, q):
    r"""
    Lorentzian distance:

    .. math::

        \sum{\ln(1+|P_i-Q_i|)}
    """
    return np.sum(np.log(1 + np.abs(p - q)))


def penrose_shape_distance(p, q):
    r"""
    Penrose shape distance:

    .. math::

        \sqrt{\sum((P_i-\bar{P})-(Q_i-\bar{Q}))^2}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return np.sqrt(np.sum(np.power((p - p_avg) - (q - q_avg), 2)))


def clark_distance(p, q):
    r"""
    Clark distance:

    .. math::

        (\frac{1}{N}\sum(\frac{P_i-Q_i}{|P_i|+|Q_i|})^2)^\frac{1}{2}
    """
    n = np.sum(p > 0)
    return np.sqrt(1 / n * np.sum(np.power((p - q) / (np.abs(p) + np.abs(q)), 2)))


def hellinger_distance(p, q):
    r"""
    Hellinger distance:

    .. math::

        \sqrt{2\sum(\sqrt{\frac{P_i}{\bar{P}}}-\sqrt{\frac{Q_i}{\bar{Q}}})^2}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return np.sqrt(2 * np.sum(np.power(np.sqrt(p / p_avg) - np.sqrt(q / q_avg), 2)))


def whittaker_index_of_association_distance(p, q):
    r"""
    Whittaker index of association distance:

    .. math::

        \frac{1}{2}\sum|\frac{P_i}{\bar{P}}-\frac{Q_i}{\bar{Q}}|
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return 1 / 2 * np.sum(np.abs(p / p_avg - q / q_avg))


def symmetric_chi_squared_distance(p, q):
    r"""
    Symmetric χ2 distance:

    .. math::

        \sqrt{\sum{\frac{\bar{P}+\bar{Q}}{N(\bar{P}+\bar{Q})^2}\frac{(P_i\bar{Q}-Q_i\bar{P})^2}{P_i+Q_i}\ }}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    n = np.sum(p > 0)

    d1 = (p_avg + q_avg) / (n * np.power(p_avg + q_avg, 2))
    return np.sqrt(d1 * np.sum(np.power(p * q_avg - q * p_avg, 2) / (p + q)))


def pearson_correlation_distance(p, q):
    r"""
    Pearson/Spearman Correlation Coefficient:

    .. math::

        \frac{\sum[(Q_i-\bar{Q})(P_i-\bar{P})]}{\sqrt{\sum(Q_i-\bar{Q})^2\sum(P_i-\bar{P})^2}}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)

    x = np.sum((q - q_avg) * (p - p_avg))
    y = np.sqrt(np.sum(np.power(q - q_avg, 2)) * np.sum(np.power(p - p_avg, 2)))

    if x == 0 and y == 0:
        return 0.0
    else:
        return -x / y


def improved_similarity_distance(p, q):
    r"""
    Improved Similarity Index:

    .. math::

        \sqrt{\frac{1}{N}\sum\{\frac{P_i-Q_i}{P_i+Q_i}\}^2}
    """
    n = np.sum(p > 0)
    return np.sqrt(1 / n * np.sum(np.power((p - q) / (p + q), 2)))


def absolute_value_distance(p, q):
    r"""
    Absolute Value Distance:

    .. math::

        \frac { \sum(|Q_i-P_i|)}{\sum P_i}

    """
    dist = np.sum(np.abs(q - p)) / np.sum(p)
    return dist


def dot_product_distance(p, q):
    r"""
    Dot product distance:

    .. math::

        1 - \sqrt{\frac{(\sum{Q_iP_i})^2}{\sum{Q_i^2\sum P_i^2}}}
    """
    score = np.power(np.sum(q * p), 2) / (
        np.sum(np.power(q, 2)) * np.sum(np.power(p, 2))
    )
    return 1 - np.sqrt(score)


def cosine_distance(p, q):
    r"""
    Cosine distance, it gives the same result as the dot product.

    .. math::

        1 - \sqrt{\frac{(\sum{Q_iP_i})^2}{\sum{Q_i^2\sum P_i^2}}}
    """
    return dot_product_distance(p, q)


def perc_peaks_common_distance(p, q):
    "max is 1"

    matched_peaks = len(np.where(p * q > 0)[0])
    peaks_p = len(np.where(p > 0)[0])
    peaks_q = len(np.where(q > 0)[0])

    return 1 - min(matched_peaks / peaks_p, matched_peaks / peaks_q)


def rbf_distance(p, q):
    """
    flexible norm constant implemented
    lim_1 and lim_2 will be passed thru kernel function to get max distance for this length
    In the case that the length in common is 0, return 1 for distance
    """

    if len(p) == 1:
        return 1

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    lim_2[1] += 1

    return (1 - pk([p], [q], metric="rbf")[0][0]) / (
        1 - pk([lim_1], [lim_2], metric="rbf")[0][0]
    )


def chi2_distance(p, q):
    "max is always 1, no need for normalization constant"

    return 1 - pk([p], [q], metric="chi2")[0][0]


def additive_chi2_distance(p, q):
    "max dist is always -2, no need for normalization constant"

    return pk([p], [q], metric="additive_chi2")[0][0] / -2


def linear_distance(p, q):
    "max is always 1, no need for normalization constant"

    return 1 - pk([p], [q], metric="linear")[0][0]


def reverse_distance(p, q, metric):

    p, q = _select_common_peaks(p, q)
    if np.sum(p) == 0:
        return 1
    else:
        return eval(f"{metric}_distance({p},{q})")


def max_distance(p, q, metric):

    return max(
        eval(f"reverse_distance({p},{q},{metric})"),
        eval(f"reverse_distance({q},{p},{metric})"),
    )


def min_distance(p, q, metric):

    return min(
        eval(f"reverse_distance({p},{q},{metric})"),
        eval(f"reverse_distance({q},{p},{metric})"),
    )


def ave_distance(p, q, metric):

    return (
        eval(f"reverse_distance({p},{q},{metric})")
        + eval(f"reverse_distance({q},{p},{metric})")
    ) / 2


def minkowski_distance(p_, q):
    """
    max is 1.18
    """

    if len(p_) == 1:
        return 1

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p_))
    lim_2 = np.zeros(len(p_))

    lim_1[0] += 1
    lim_2[1] += 1

    return dist.minkowski(p_, q, p=4) / dist.minkowski(lim_1, lim_2, p=4)


def correlation_distance(p, q):
    """
    max is 2
    """

    if len(p) < 2:
        return 1

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    lim_2[1] += 1

    return dist.correlation(p, q) / dist.correlation(lim_1, lim_2)


def jensenshannon_distance(p, q):
    """
    max is 0.82
    """

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    lim_2[1] += 1

    return dist.jensenshannon(p, q) / dist.jensenshannon(lim_1, lim_2)


def sqeuclidean_distance(p, q):
    """
    max is 2
    """

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    lim_2[1] += 1

    return dist.sqeuclidean(p, q) / dist.sqeuclidian(lim_1, lim_2)


def gini_distance(p, q):

    matched = (p + q) / 2

    # create zero arrays for normalization constant
    lim_1 = np.zeros(len(p))
    lim_2 = np.zeros(len(p))

    lim_1[0] += 1
    lim_2[1] += 1

    lim_matched = (lim_1 + lim_2) / 2

    return (
        (matched @ (1 - matched))
        - (p @ (1 - p))
        - (q @ (1 - q)) / (lim_matched @ (1 - lim_matched))
        - (lim_1 @ (1 - lim_1))
        - (lim_2 @ (1 - lim_2))
    )


def standardize_spectrum(spectrum):
    """
    Sort spectrum by m/z, normalize spectrum to have intensity sum = 1.
    """
    spectrum = spectrum[np.argsort(spectrum[:, 0])]
    intensity_sum = np.sum(spectrum[:, 1])
    if intensity_sum > 0:
        spectrum[:, 1] /= intensity_sum
    return spectrum


def clean_spectrum(
    spectrum,
    max_mz: float = None,
    noise_removal: float = 0.01,
    ms2_da: float = 0.05,
    ms2_ppm: float = None,
) -> np.ndarray:
    """
    Clean the spectrum with the following procedures:
    1. Remove ions have m/z higher than a given m/z (defined as max_mz).
    2. Centroid peaks by merging peaks within a given m/z (defined as ms2_da or ms2_ppm).
    3. Remove ions have intensity lower than max intensity * fixed value (defined as noise_removal)


    :param spectrum: The input spectrum, need to be in 2-D list or 2-D numpy array
    :param max_mz: The ions with m/z higher than max_mz will be removed.
    :param noise_removal: The ions with intensity lower than max ion's intensity * noise_removal will be removed.
    :param ms2_da: The MS/MS tolerance in Da.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    If both ms2_da and ms2_ppm is given, ms2_da will be used.
    """
    # Check parameter
    spectrum = check_spectrum(spectrum)
    if ms2_da is None and ms2_ppm is None:
        raise RuntimeError("MS2 tolerance need to be set!")

    # 1. Remove the precursor ions
    if max_mz is not None:
        spectrum = spectrum[spectrum[:, 0] <= max_mz]

    # 2. Centroid peaks
    spectrum = spectrum[np.argsort(spectrum[:, 0])]
    spectrum = centroid_spec(spectrum, ms2_da=ms2_da, ms2_ppm=ms2_ppm)

    # 3. Remove noise ions
    if noise_removal is not None and spectrum.shape[0] > 0:
        max_intensity = np.max(spectrum[:, 1])
        spectrum = spectrum[spectrum[:, 1] >= max_intensity * noise_removal]

    # 4. Standardize the spectrum.
    spectrum = standardize_spectrum(spectrum)
    return spectrum


def centroid_spec(spec, ms2_ppm=None, ms2_da=None):
    try:
        return tools_fast.centroid_spec(spec, ms2_ppm, ms2_da)
    except Exception as e:
        pass
    """
    If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    """
    # Fast check is the spectrum need centroid.
    mz_array = spec[:, 0]
    need_centroid = 0
    if mz_array.shape[0] > 1:
        mz_delta = mz_array[1:] - mz_array[:-1]
        if ms2_da is not None:
            if np.min(mz_delta) <= ms2_da:
                need_centroid = 1
        else:
            if np.min(mz_delta / mz_array[1:] * 1e6) <= ms2_ppm:
                need_centroid = 1

    if need_centroid:
        intensity_order = np.argsort(-spec[:, 1])
        spec_new = []
        for i in intensity_order:
            if ms2_da is None:
                if ms2_ppm is None:
                    raise RuntimeError("MS2 tolerance not defined.")
                else:
                    mz_delta_allowed = ms2_ppm * 1e-6 * spec[i, 0]
            else:
                mz_delta_allowed = ms2_da

            if spec[i, 1] > 0:
                # Find left board for current peak
                i_left = i - 1
                while i_left >= 0:
                    mz_delta_left = spec[i, 0] - spec[i_left, 0]
                    if mz_delta_left <= mz_delta_allowed:
                        i_left -= 1
                    else:
                        break
                i_left += 1

                # Find right board for current peak
                i_right = i + 1
                while i_right < spec.shape[0]:
                    mz_delta_right = spec[i_right, 0] - spec[i, 0]
                    if mz_delta_right <= mz_delta_allowed:
                        i_right += 1
                    else:
                        break

                # Merge those peaks
                intensity_sum = np.sum(spec[i_left:i_right, 1])
                intensity_weighted_sum = np.sum(
                    spec[i_left:i_right, 0] * spec[i_left:i_right, 1]
                )

                spec_new.append([intensity_weighted_sum / intensity_sum, intensity_sum])
                spec[i_left:i_right, 1] = 0

        spec_new = np.array(spec_new)
        # Sort by m/z
        spec_new = spec_new[np.argsort(spec_new[:, 0])]
        return spec_new
    else:
        return spec


def match_peaks_in_spectra(spec_a, spec_b, ms2_ppm=None, ms2_da=None):
    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :return: list. Each element in the list is a list contain three elements:
                              m/z, intensity from spec 1; intensity from spec 2.
    """
    try:
        return tools_fast.match_spectrum(spec_a, spec_b, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
    except Exception as e:
        pass

    a = 0
    b = 0

    spec_merged = []
    peak_b_int = 0.0

    while a < spec_a.shape[0] and b < spec_b.shape[0]:
        if ms2_da is None:
            ms2_da = ms2_ppm * spec_a[a, 0] * 1e6
        mass_delta = spec_a[a, 0] - spec_b[b, 0]

        if mass_delta < -ms2_da:
            # Peak only existed in spec a.
            spec_merged.append([spec_a[a, 0], spec_a[a, 1], peak_b_int])
            peak_b_int = 0.0
            a += 1
        elif mass_delta > ms2_da:
            # Peak only existed in spec b.
            spec_merged.append([spec_b[b, 0], 0.0, spec_b[b, 1]])
            b += 1
        else:
            # Peak existed in both spec.
            peak_b_int += spec_b[b, 1]
            b += 1

    if peak_b_int > 0.0:
        spec_merged.append([spec_a[a, 0], spec_a[a, 1], peak_b_int])
        peak_b_int = 0.0
        a += 1

    if b < spec_b.shape[0]:
        spec_merged += [[x[0], 0.0, x[1]] for x in spec_b[b:]]

    if a < spec_a.shape[0]:
        spec_merged += [[x[0], x[1], 0.0] for x in spec_a[a:]]

    if spec_merged:
        spec_merged = np.array(spec_merged, dtype=np.float64)
    else:
        spec_merged = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    return spec_merged


def match_peaks_with_mz_info_in_spectra(spec_a, spec_b, ms2_ppm=None, ms2_da=None):
    """
    Match two spectra, find common peaks. If both ms2_ppm and ms2_da is defined, ms2_da will be used.
    :return: list. Each element in the list is a list contain three elements:
                              m/z from spec 1; intensity from spec 1; m/z from spec 2; intensity from spec 2.
    """
    a = 0
    b = 0

    spec_merged = []
    peak_b_mz = 0.0
    peak_b_int = 0.0

    while a < spec_a.shape[0] and b < spec_b.shape[0]:
        mass_delta_ppm = (spec_a[a, 0] - spec_b[b, 0]) / spec_a[a, 0] * 1e6
        if ms2_da is not None:
            ms2_ppm = ms2_da / spec_a[a, 0] * 1e6
        if mass_delta_ppm < -ms2_ppm:
            # Peak only existed in spec a.
            spec_merged.append([spec_a[a, 0], spec_a[a, 1], peak_b_mz, peak_b_int])
            peak_b_mz = 0.0
            peak_b_int = 0.0
            a += 1
        elif mass_delta_ppm > ms2_ppm:
            # Peak only existed in spec b.
            spec_merged.append([0.0, 0.0, spec_b[b, 0], spec_b[b, 1]])
            b += 1
        else:
            # Peak existed in both spec.
            peak_b_mz = ((peak_b_mz * peak_b_int) + (spec_b[b, 0] * spec_b[b, 1])) / (
                peak_b_int + spec_b[b, 1]
            )
            peak_b_int += spec_b[b, 1]
            b += 1

    if peak_b_int > 0.0:
        spec_merged.append([spec_a[a, 0], spec_a[a, 1], peak_b_mz, peak_b_int])
        peak_b_mz = 0.0
        peak_b_int = 0.0
        a += 1

    if b < spec_b.shape[0]:
        spec_merged += [[0.0, 0.0, x[0], x[1]] for x in spec_b[b:]]

    if a < spec_a.shape[0]:
        spec_merged += [[x[0], x[1], 0.0, 0.0] for x in spec_a[a:]]

    if spec_merged:
        spec_merged = np.array(spec_merged, dtype=np.float64)
    else:
        spec_merged = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    return spec_merged


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


methods_range = {
    "entropy": [0, np.log(4)],
    "entropy_mod_3": [0, np.log(4)],
    "max_entropy": [0, np.log(4)],
    "min_entropy": [0, np.log(4)],
    "entropy_reverse": [0, np.log(4)],
    "unweighted_entropy": [0, np.log(4)],
    "unweighted_entropy_mod_3": [0, np.log(4)],
    "unweighted_entropy_reverse": [0, np.log(4)],
    "max_unweighted_entropy": [0, np.log(4)],
    "min_unweighted_entropy": [0, np.log(4)],
    "absolute_value": [0, 2],
    "avg_l": [0, 1.5],
    "bhattacharya_1": [0, np.arccos(0) ** 2],
    "bhattacharya_1_mod_3": [0, np.arccos(0) ** 2],
    "bhattacharya_1_reverse": [0, np.arccos(0) ** 2],
    "max_bhattacharya_1": [0, np.arccos(0) ** 2],
    "min_bhattacharya_1": [0, np.arccos(0) ** 2],
    "bhattacharya_2_mod_3": [0, np.inf],
    "bhattacharya_2_reverse": [0, np.inf],
    "min_bhattacharya_2": [0, np.inf],
    "max_bhattacharya_2": [0, np.inf],
    "bhattacharya_2": [0, np.inf],
    "canberra": [0, np.inf],
    "canberra_mod_3": [0, np.inf],
    "canberra_reverse": [0, np.inf],
    "max_canberra": [0, np.inf],
    "min_canberra": [0, np.inf],
    "clark": [0, np.inf],
    "divergence": [0, np.inf],
    "euclidean": [0, np.sqrt(2)],
    "hellinger": [0, np.inf],
    "hellinger_mod_3": [0, np.inf],
    "hellinger_reverse": [0, np.inf],
    "max_hellinger": [0, np.inf],
    "min_hellinger": [0, np.inf],
    "improved_similarity": [0, np.inf],
    "lorentzian": [0, np.inf],
    "weighted_lorentzian": [0, np.inf],
    "lorentzian_mod_3": [0, np.inf],
    "lorentzian_reverse": [0, np.inf],
    "max_lorentzian": [0, np.inf],
    "min_lorentzian": [0, np.inf],
    "manhattan": [0, 2],
    "manhattan_mod_3": [0, 2],
    "manhattan_reverse": [0, 2],
    "max_manhattan": [0, 2],
    "min_manhattan": [0, 2],
    "matusita": [0, np.sqrt(2)],
    "matusita_mod_3": [0, np.sqrt(2)],
    "matusita_reverse": [0, np.sqrt(2)],
    "max_matusita": [0, np.sqrt(2)],
    "min_matusita": [0, np.sqrt(2)],
    "mean_character": [0, 2],
    "motyka": [-0.5, 0],
    "ms_for_id": [-np.inf, 0],
    "ms_for_id_mod_3": [-np.inf, 0],
    "ms_for_id_reverse": [-np.inf, 0],
    "max_ms_for_id": [-np.inf, 0],
    "min_ms_for_id_v1": [0, np.inf],
    "pearson_correlation": [-1, 1],
    "penrose_shape": [0, np.sqrt(2)],
    "penrose_size": [0, np.inf],
    "probabilistic_symmetric_chi_squared": [0, 1],
    "probabilistic_symmetric_chi_squared_mod_3": [0, 1],
    "probabilistic_symmetric_chi_squared_reverse": [0, 1],
    "max_probabilistic_symmetric_chi_squared": [0, 1],
    "min_probabilistic_symmetric_chi_squared": [0, 1],
    "similarity_index": [0, np.inf],
    "squared_chord": [0, 2],
    "squared_chord_mod_3": [0, 2],
    "squared_chord_reverse": [0, 2],
    "max_squared_chord": [0, 2],
    "min_squared_chord": [0, 2],
    "squared_euclidean": [0, 2],
    "symmetric_chi_squared": [0, 0.5 * np.sqrt(2)],
    "vicis_symmetric_chi_squared_3": [0, 2],
    "vicis_symmetric_chi_squared_3_mod_3": [0, 2],
    "vicis_symmetric_chi_squared_3_reverse": [0, 2],
    "max_vicis_symmetric_chi_squared_3": [0, 2],
    "min_vicis_symmetric_chi_squared_3": [0, 2],
    "wave_hedges": [0, np.inf],
    "whittaker_index_of_association": [0, np.inf],
    "max_revdot_distance": [0, 1],
    "perc_peaks_in_common_distance": [0, 1],
    "rbf_distance": [0, 1],
    "chi2_distance": [0, 1],
    "cosine_kernel_distance": [0, 1],
    "laplacian_distance": [0, 1],
    "minkowski_distance": [0, 1],
    "correlation_distance": [0, 1],
    "jensenshannon_distance": [0, 1],
    "sqeuclidean_distance": [0, 1],
    "min_revdot_distance": [0, 1],
    "rbf_mod_distance": [0, 1],
    "chi2_mod_distance": [0, 1],
    "cosine_kernel_mod_distance": [0, 1],
    "laplacian_mod_distance": [0, 1],
    "gini": [0, 1],
    "gini_mod": [0, 1],
    "l2": [0, 1],
    "l2_mod": [0, 1],
    "common_mass": [0, 1],
    "entropy_mod": [0, 1],
    "cross_ent": [0, 1],
    "cross_ent_mod": [0, 1],
    "binary_cross_ent": [0, 1],
    "binary_cross_ent_mod": [0, 1],
    "kl": [0, 1],
    "kl_mod": [0, 1],
    "kl_mod_2": [0, np.inf],
    "rbf_mod_2": [0, np.inf],
    "chi2_mod_2": [0, np.inf],
    "cosine_kernel_mod_2": [0, np.inf],
    "laplacian_mod_2": [0, np.inf],
    "gini_mod_2": [0, np.inf],
    "l2_mod_2": [0, np.inf],
    "entropy_mod_2": [0, np.inf],
    "cross_ent_mod_2": [0, np.inf],
    "binary_cross_ent_mod_2": [0, np.inf],
    "kl_mod_2": [0, np.inf],
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
    :param method: Supported methods:
            "entropy", "unweighted_entropy", "euclidean", "manhattan", "chebyshev", "squared_euclidean", "fidelity", \
            "matusita", "squared_chord", "bhattacharya_1", "bhattacharya_2", "harmonic_mean", \
            "probabilistic_symmetric_chi_squared", "ruzicka", "roberts", "intersection", \
            "motyka", "canberra", "baroni_urbani_buser", "penrose_size", "mean_character", "lorentzian",\
            "penrose_shape", "clark", "hellinger", "whittaker_index_of_association", "symmetric_chi_squared", \
            "pearson_correlation", "improved_similarity", "absolute_value", "dot_product", "dot_product_reverse", \
            "spectral_contrast_angle", "wave_hedges", "jaccard", "dice", "inner_product", "divergence", \
            "avg_l", "vicis_symmetric_chi_squared_3", "ms_for_id_v1", "ms_for_id", "weighted_dot_product"
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
    :param method: Supported methods:
            "entropy", "unweighted_entropy", "euclidean", "manhattan", "chebyshev", "squared_euclidean", "fidelity", \
            "matusita", "squared_chord", "bhattacharya_1", "bhattacharya_2", "harmonic_mean", \
            "probabilistic_symmetric_chi_squared", "ruzicka", "roberts", "intersection", \
            "motyka", "canberra", "baroni_urbani_buser", "penrose_size", "mean_character", "lorentzian",\
            "penrose_shape", "clark", "hellinger", "whittaker_index_of_association", "symmetric_chi_squared", \
            "pearson_correlation", "improved_similarity", "absolute_value", "dot_product", "dot_product_reverse", \
            "spectral_contrast_angle", "wave_hedges", "jaccard", "dice", "inner_product", "divergence", \
            "avg_l", "vicis_symmetric_chi_squared_3", "ms_for_id_v1", "ms_for_id", "weighted_dot_product"
    :param ms2_ppm: The MS/MS tolerance in ppm.
    :param ms2_da: The MS/MS tolerance in Da.
    :param need_clean_spectra: Normalize spectra before comparing, required for not normalized spectrum.
    :param need_normalize_result: Normalize the result into [0,1].
    :return: Distance between two spectra
    """

    if ms2_ppm is None and ms2_da is None:
        raise ValueError("MS2 tolerance need to be defined!")

    spectrum_query = np.asarray(spectrum_query, dtype=np.float32)
    spectrum_library = np.asarray(spectrum_library, dtype=np.float32)
    if need_clean_spectra:
        spectrum_query = clean_spectrum(spectrum_query, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        spectrum_library = clean_spectrum(
            spectrum_library, ms2_ppm=ms2_ppm, ms2_da=ms2_da
        )

    # Calculate similarity
    if spectrum_query.shape[0] > 0 and spectrum_library.shape[0] > 0:
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
            dist = f(spectrum_query, spectrum_library, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
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


def all_distance(
    spectrum_query: Union[list, np.ndarray],
    spectrum_library: Union[list, np.ndarray],
    ms2_ppm: float = None,
    ms2_da: float = None,
    need_clean_spectra: bool = True,
    need_normalize_result: bool = True,
) -> dict:
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
        raise ValueError("MS2 tolerance need to be defined!")
    spectrum_query = np.asarray(spectrum_query, dtype=np.float32)
    spectrum_library = np.asarray(spectrum_library, dtype=np.float32)
    if need_clean_spectra:
        spectrum_query = clean_spectrum(spectrum_query, ms2_ppm=ms2_ppm, ms2_da=ms2_da)
        spectrum_library = clean_spectrum(
            spectrum_library, ms2_ppm=ms2_ppm, ms2_da=ms2_da
        )

    # Calculate similarity
    result = {}
    if spectrum_query.shape[0] > 0 and spectrum_library.shape[0] > 0:
        spec_matched = match_peaks_in_spectra(
            spec_a=spectrum_query,
            spec_b=spectrum_library,
            ms2_ppm=ms2_ppm,
            ms2_da=ms2_da,
        )
        for method in methods_name:
            function_name = method + "_distance"
            if hasattr(math_distance, function_name):
                f = getattr(math_distance, function_name)
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
            result[method] = dist

    else:
        for method in methods_name:
            if need_normalize_result:
                result[method] = 1
            else:
                result[method] = np.inf
    return result


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
    if methods:
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
    else:
        return all_distance(
            spectrum_query=spectrum_query,
            spectrum_library=spectrum_library,
            need_clean_spectra=need_clean_spectra,
            need_normalize_result=need_normalize_result,
            ms2_ppm=ms2_ppm,
            ms2_da=ms2_da,
        )
