import statistics
from typing import Any, Callable

import joblib
import numpy as np
import roughpy as rp
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def stream_normalise_mean_and_range(stream: np.array) -> np.array:
    """
    Normalise the input stream to have mean 0 and be in [-1,1] in each channel.

    Parameters
    ----------
    stream : np.array
        A two-dimensional numpy array of shape [length x dimension]

    Returns
    -------
    np.array
        A copy of the input stream, retaining the same shape, but scaled
        to have mean 0 and coordinates/channels in [-1,1]
    """
    # compute the range of each channel
    range = stream.ptp(axis=0)
    # replace 0s with 1s to avoid division by 0
    range[range == 0] = 1
    # scale the stream
    return (stream - stream.mean(axis=0)) / range


def stream_normalise_mean_and_std(stream: np.array) -> np.array:
    """
    Normalise the input stream to have mean 0 and standard deviation 1 in each channel.

    Parameters
    ----------
    stream : np.array
        A two-dimensional numpy array of shape [length x dimension]

    Returns
    -------
    np.array
        A copy of the input stream, retaining the same shape, but scaled
        to have mean 0 and standard deviation 1.0
    """
    # compute the standard deviation of each channel
    std = stream.std(axis=0)
    # replace 0s with 1s to avoid division by 0
    std[std == 0] = 1
    # scale the stream
    return (stream - stream.mean(axis=0)) / std


def compute_signature(stream: np.array, depth: int) -> np.array:
    """
    Given a stream of data, compute the signature of the stream truncated
    to the given depth using RoughPy.

    RoughPy works with increments of the stream, and we assume here that
    stream is the raw stream so incremements are computed within this function.

    Parameters
    ----------
    stream : np.array
        A two-dimensional numpy array of shape [length x dimension]
    depth : int
        The depth to which the signature should be truncated

    Returns
    -------
    np.array
        A one-dimensional numpy array of length 1+dim+dim^2+...+dim^depth
        containing the signature of the stream truncated to the given depth
    """
    # define context for RoughPy increment stream
    context = rp.get_context(width=stream.shape[1], depth=depth, coeffs=rp.DPReal)
    # compute increments of the stream
    increments = np.diff(stream, axis=0)
    # define RoughPy increment stream object
    lie_incremement_stream = rp.LieIncrementStream.from_increments(
        increments, ctx=context
    )
    # compute signature and convert to numpy array
    return np.array(lie_incremement_stream.signature())


def sig_scale_depth_ratio(
    sig: np.array,
    dim: int,
    depth: int,
    scale_factor: float,
) -> np.array:
    """
    The returned output is a numpy array of the same shape as the input signature, sig.
    Each entry in the returned array is the corresponding entry in sig multiplied by r^k,
    where k is the level of sig to which the entry belongs.

    If the input depth is 0, an error message is returned since the scaling will always
    leave the level zero entry of a signature as 1.

    Parameters
    ----------
    sig : np.array
        A one-dimensional numpy array of length 1+dim+dim^2+...+dim^depth corresponding
        to the signature of a stream of dimension dim and truncated to level depth.
    dim : int
        Dimension of the input stream
    depth : int
        Level to which the input signature is truncated
    scale_factor : float
        The scaling factor to be applied to the signature terms

    Returns
    -------
    np.array
        A one-dimensional numpy array of length 1+dim+dim^2+...+dim^depth.
        Each entry is the corresponding entry in sig multiplied by r^k, where k
        is the level of sig to which the entry belongs.
    """
    # compute the number of signature terms for each depth/level of the signature
    sig_terms_per_depth = [dim**d for d in range(depth + 1)]
    if depth == 0:
        raise ValueError(
            "Depth 0 term of a signature is always 1 and will not be changed by scaling"
        )
    elif sig.shape != (sum(sig_terms_per_depth),):
        raise ValueError(
            f"sig is not the correct shape for a signature of dimension {dim} and depth {depth}."
            f"Expected shape: 1+dim+dim^2+...+dim^depth = {sum(sig_terms_per_depth)},"
            f"but got shape: {sig.shape}."
        )
    else:
        # scale the signature terms at each depth
        for d in range(depth + 1):
            sig[sum(sig_terms_per_depth[:d]) : sum(sig_terms_per_depth[: d + 1])] *= (
                scale_factor**d
            )

        return sig


def mnist_train_data(
    data_directory: str, cpu_number: int
) -> tuple[list[np.array], list[np.array], list[int]]:
    """
    Read the training data from the specified directory.

    Parameters
    ----------
    data_directory : str
        The directory containing the MNIST sequences data
    cpu_number : int
        The number of cpus to use for parallelisation

    Returns
    -------
    tuple[list[np.array], list[np.array], list[int]]
        The first item is a list of the streams formed by the 2D
        points data, the second is a list of streams of the 3D input data recording
        in the first two entries the change in x and y coordinates from the previous
        point and when the pen leaves the paper in the third entry. The fourth
        coordinate used to indicate when one figure ends and the next begins is
        ignored in this extraction. See the MNIST sequences project website for full
        details. The third returned list is a list of integer labels marking the
        number corresponding to each sequence.
    """

    def func_A(data_directory: str, k: int) -> np.array:
        return np.loadtxt(
            f"./{data_directory}/sequences/trainimg-{k}-points.txt",
            delimiter=",",
            skiprows=1,
        )[:-1, :]

    def func_B(data_directory: str, k: int) -> np.array:
        return np.loadtxt(
            f"./{data_directory}/sequences/trainimg-{k}-inputdata.txt",
            delimiter=" ",
            usecols=np.arange(0, 3),
        )[:-1, :]

    def func_C(data_directory: str, k: int) -> int:
        return statistics.mode(
            np.argmax(
                np.loadtxt(
                    f"./{data_directory}/sequences/trainimg-{k}-targetdata.txt",
                    delimiter=" ",
                ),
                axis=1,
            )
        )

    A = Parallel(n_jobs=cpu_number)(
        [delayed(func_A)(data_directory, j) for j in range(60000)]
    )
    B = Parallel(n_jobs=cpu_number)(
        [delayed(func_B)(data_directory, j) for j in range(60000)]
    )
    C = Parallel(n_jobs=cpu_number)(
        [delayed(func_C)(data_directory, j) for j in range(60000)]
    )

    return A, B, C


def mnist_test_data(
    data_directory: str, cpu_number: int
) -> tuple[list[np.array], list[np.array], list[int]]:
    """
    Read the test data from the specified directory.

    Parameters
    ----------
    data_directory : str
        The directory containing the MNIST sequences data
    cpu_number : int
        The number of cpus to use for parallelisation

    Returns
    -------
    tuple[list[np.array], list[np.array], list[int]]
        The first is a list of the streams formed by the 2D
        points data, the second is a list of streams of the 3D input data recording
        in the first two entries the change in x and y coordinates from the previous
        point and when the pen leaves the paper in the third entry. The fourth
        coordinate used to indicate when one figure ends and the next begins is
        ignored in this extraction. See the MNIST sequences project website for full
        details. The third returned list is the integer labels marking the number
        corresponding to each sequence.
    """

    def func_A(data_directory: str, k: int) -> np.array:
        return np.loadtxt(
            f"./{data_directory}/sequences/testimg-{k}-points.txt",
            delimiter=",",
            skiprows=1,
        )[:-1, :]

    def func_B(data_directory: str, k: int) -> np.array:
        return np.loadtxt(
            f"./{data_directory}/sequences/testimg-{k}-inputdata.txt",
            delimiter=" ",
            usecols=np.arange(0, 3),
        )[:-1, :]

    def func_C(data_directory: str, k: int) -> int:
        return statistics.mode(
            np.argmax(
                np.loadtxt(
                    f"./{data_directory}/sequences/testimg-{k}-targetdata.txt",
                    delimiter=" ",
                ),
                axis=1,
            )
        )

    A = Parallel(n_jobs=cpu_number)(
        [delayed(func_A)(data_directory, j) for j in range(10000)]
    )
    B = Parallel(n_jobs=cpu_number)(
        [delayed(func_B)(data_directory, j) for j in range(10000)]
    )
    C = Parallel(n_jobs=cpu_number)(
        [delayed(func_C)(data_directory, j) for j in range(10000)]
    )

    return A, B, C


def ridge_learn(
    scale_factor: float,
    dim: int,
    depth: int,
    data: list[np.array],
    labels: list[int],
    CV: Any = None,
    reg: list[np.array] = [np.array((0.1, 1, 10))],
    cpu_number: int = 1,
) -> tuple[float, RidgeClassifierCV, float]:
    """
    Fit a RidgeClassifierCV model to the input data, using the given scale factor.

    The entries in the data list are scaled via the sig_scale_depth_ratio
    function, i.e. via sig_scale_depth_ratio(data, dim, depth,
    scale_factor=scale_factor), and cpu_number number of cpus are used for
    parallelisation.

    Once scaled, a sklearn GridSearchCV is run with the model set to be
    RidgeClassifierCV(), the param_grid to be {'alphas':reg} and the
    cross-validation strategy to be determined by CV.

    The selected best model is used to predict the labels for the appropriately
    scaled data, and the accuracy_score of the predicted labels compared to the
    actual labels is computed.

    The returned output is a list composed of the scale_factor used, the model
    selected during the GridSearch, and the accuracy_score achieved by the
    selected model.

    Parameters
    ----------
    scale_factor : float
        The scaling factor to be applied to the signature terms
    dim : int
        Dimension of the input stream
    depth : int
        Level to which the input signature is truncated
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]
        containing the signature of the stream
    labels : list[int]
        List of integer labels marking the number corresponding to each sequence
    CV : Any, optional
        Must be an int, cross-validation generator or an iterable.
        See scikit-learn documentation for more details, by default None
    reg : list[np.array], optional
        List of possible alpha inputs for a RidgeClassifierCV model,
        by default [np.array((0.1, 1, 10))]
    cpu_number : int, optional
        The number of cpus to use for parallelisation, by default 1

    Returns
    -------
    tuple[float, RidgeClassifierCV, float]
        A tuple containing the scale_factor used, the model selected during the
        GridSearch, and the accuracy_score achieved by the selected model
    """
    if depth == 0:
        return print(
            "Error: Depth 0 term of signature is always 1 and will not change under scaling"
        )
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    else:
        ridge = RidgeClassifierCV()
        tuned_params = {"alphas": reg}
        Q = Parallel(n_jobs=cpu_number)(
            [
                delayed(sig_scale_depth_ratio)(data[k], dim, depth, scale_factor)
                for k in range(len(data))
            ]
        )
        model = GridSearchCV(
            estimator=ridge, param_grid=tuned_params, cv=CV, n_jobs=cpu_number
        )
        model.fit(Q, labels)
        best_model = model.best_estimator_
        preds = best_model.predict(Q)
        acc = accuracy_score(preds, labels)
        return scale_factor, best_model, acc


def ridge_scale_learn(
    scale_factors: list[float],
    dim: int,
    depth: int,
    data: list[np.array],
    labels: list[int],
    CV: Any = None,
    reg: list[np.array] = [np.array((0.1, 1, 10))],
    cpu_number_one: int = 1,
    cpu_number_two: int = 1,
) -> tuple[
    list[tuple[float, RidgeClassifierCV, float]],
    tuple[float, RidgeClassifierCV, float],
]:
    """
    Fit a RidgeClassifierCV model to the input data, using the given scale factors.

    The returned outputs are a list of the results of ridge_learn(r, dim, depth,
    data, labels, CV, reg) for each entry r in scale_factors, and a separate
    recording of the output corresponding to the entry r in scale_factors for
    which the best accuracy_score is achieved The integer cpu_number_one
    determines how many cpus are used for parallelisation over the scale_factors,
    whilst cpu_number_two determines the number of cpus used for parallelisation
    during the call of the function ridge_learn for each scale_factor considered.

    Parameters
    ----------
    scale_factors : list[float]
        List of scaling factors to be applied to the signature terms
    dim : int
        Dimension of the input stream
    depth : int
        Level to which the input signature is truncated
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]
        containing the signature of the stream
    labels : list[int]
        List of integer labels marking the number corresponding to each sequence
    CV : Any, optional
        Must be an int, cross-validation generator or an iterable.
        See scikit-learn documentation for more details, by default None
    reg : list[np.array], optional
        List of possible alpha inputs for a RidgeClassifierCV model,
        by default [np.array((0.1, 1, 10))]
    cpu_number_one : int, optional
        The number of cpus to use for parallelisation over the scale_factors,
        by default 1
    cpu_number_two : int, optional
        The number of cpus to use for parallelisation during the call of the
        function ridge_learn for each scale_factor considered, by default 1

    Returns
    -------
    tuple[list[tuple[float, RidgeClassifierCV, float]], tuple[float, RidgeClassifierCV, float]]
        A tuple containing a list of the results of ridge_learn(r, dim, depth,
        data, labels, CV, reg) for each entry r in scale_factors, and a separate
        recording of the output corresponding to the entry r in scale_factors for
        which the best accuracy_score is achieved
    """
    if depth == 0:
        return print(
            "Error: Depth 0 term of signature is always 1 and will not change under scaling"
        )
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    if cpu_number_one + cpu_number_two > joblib.cpu_count() and (
        cpu_number_one != 1 or cpu_number_two != 1
    ):
        return print("Error: Tried to allocate more than the number of available cpus.")
    else:
        results = Parallel(n_jobs=cpu_number_one)(
            [
                delayed(ridge_learn)(
                    r, dim, depth, data, labels, CV, reg, cpu_number=cpu_number_two
                )
                for r in scale_factors
            ]
        )
        best = np.argmax(np.delete(results, 1, axis=1), axis=0)[1]
        return results, results[best]


def SVC_learn(
    scale_factor: float,
    dim: int,
    depth: int,
    data: list[np.array],
    labels: list[int],
    CV: Any = None,
    regC: list[float] = [1.0],
    reg_gamma: list[str | float] = ["scale"],
    reg_kernel: list[str | Callable] = ["rbf"],
    cpu_number: int = 1,
) -> tuple[float, SVC, float]:
    """
    Fit a SVC model to the input data, using the given scale factor.

    The entries in the data list are scaled via the sig_scale_depth_ratio
    function with scale_factor as the scale_factor, i.e.
    sig_scale_depth_ratio(data, dim, depth, scale_factor = scale_factor).

    Once scaled, a sklearn GridSearchCV is run with the model set to be
    SVC(random_state=42), the param_grid to be {'C':regC , 'gamma':reg_gamma ,
    'kernel':reg_kernel} and the cross-validation strategy to be determined by CV.

    The selected best model is used to predict the labels for the appropriately
    scaled data, and the accuracy_score of the predicted labels compared to the
    actual labels is computed. The returned output is a list composed of the
    scale_factor used, the model selected during the GridSearch, and the
    accuracy_score achieved by the selected model.

    Parameters
    ----------
    scale_factor : float
        The scaling factor to be applied to the signature terms
    dim : int
        Dimension of the input stream
    depth : int
        Level to which the input signature is truncated
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]
        containing the signature of the stream
    labels : list[int]
        List of integer labels marking the number corresponding to each sequence
    CV : Any, optional
        Must be an int, cross-validation generator or an iterable.
        See scikit-learn documentation for more details, by default None
    regC : list[float], optional
        List of possible C inputs for a SVC model, by default [1.0]
    reg_gamma : list[str  |  float], optional
        List of possible gamma value strategies for a SVC model, by default ["scale"]
    reg_kernel : list[str  |  Callable], optional
        List of possible kernels to be used in a SVC model, by default ["rbf"]
    cpu_number : int, optional
        The number of cpus to use for parallelisation, by default 1

    Returns
    -------
    tuple[float, SVC, float]
        A tuple containing the scale_factor used, the model selected during the
        GridSearch, and the accuracy_score achieved by the selected model
    """
    if depth == 0:
        return print(
            "Error: Depth 0 term of signature is always 1 and will not change under scaling"
        )
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    else:
        SVM = SVC(random_state=42)
        tuned_params = {"kernel": reg_kernel, "C": regC, "gamma": reg_gamma}
        Q = Parallel(n_jobs=cpu_number)(
            [
                delayed(sig_scale_depth_ratio)(data[k], dim, depth, scale_factor)
                for k in range(len(data))
            ]
        )
        model = GridSearchCV(
            estimator=SVM, param_grid=tuned_params, cv=CV, n_jobs=cpu_number
        )
        model.fit(Q, labels)
        best_model = model.best_estimator_
        preds = best_model.predict(Q)
        acc = accuracy_score(preds, labels)
        return scale_factor, best_model, acc


def SVC_scale_learn(
    scale_factors: list[float],
    dim: int,
    depth: int,
    data: list[np.array],
    labels: list[int],
    CV: Any = None,
    regC: list[float] = [1.0],
    reg_gamma: list[str | float] = ["scale"],
    reg_kernel: list[str | Callable] = ["rbf"],
    cpu_number_one: int = 1,
    cpu_number_two: int = 1,
) -> tuple[list[tuple[float, SVC, float]], tuple[float, SVC, float]]:
    """
    Fit a SVC model to the input data, using the given scale factors.

    The returned outputs are a list of the results of SVC_learn(r, dim, depth,
    data, labels, CV, regC, reg_gamma, reg_kernel) for each entry r in
    scale_factors, and a separate recording of the output corresponding to
    the entry r in scale_factors for which the best accuracy_score is achieved.

    The integer cpu_number_one determines how many cpus are used for
    parallelisation over the scale_factors, whilst cpu_number_two determines the
    number of cpus used for parallelisation during the call of the function
    SVC_learn for each scale_factor considered.

    Parameters
    ----------
    scale_factors : list[float]
        List of scaling factors to be applied to the signature terms
    dim : int
        Dimension of the input stream
    depth : int
        Level to which the input signature is truncated
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]
        containing the signature of the stream
    labels : list[int]
        List of integer labels marking the number corresponding to each sequence
    CV : Any, optional
        Must be an int, cross-validation generator or an iterable.
        See scikit-learn documentation for more details, by default None
    regC : list[float], optional
        List of possible C inputs for a SVC model, by default [1.0]
    reg_gamma : list[str  |  float], optional
        List of possible gamma value strategies for a SVC model, by default ["scale"]
    reg_kernel : list[str  |  Callable], optional
        List of possible kernels to be used in a SVC model, by default ["rbf"]
    cpu_number_one : int, optional
        The number of cpus to use for parallelisation over the scale_factors,
        by default 1
    cpu_number_two : int, optional
        The number of cpus to use for parallelisation during the call of the
        function ridge_learn for each scale_factor considered, by default 1

    Returns
    -------
    tuple[list[tuple[float, SVC, float]], tuple[float, SVC, float]]
        A tuple containing a list of the results of SVC_learn(r, dim, depth,
        data, labels, CV, regC, reg_gamma, reg_kernel) for each entry r in
        scale_factors, and a separate recording of the output corresponding to
        the entry r in scale_factors for which the best accuracy_score is achieved
    """
    if depth == 0:
        return print(
            "Error: Depth 0 term of signature is always 1 and will not change under scaling"
        )
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    if cpu_number_one + cpu_number_two > joblib.cpu_count() and (
        cpu_number_one != 1 or cpu_number_two != 1
    ):
        return print("Error: Tried to allocate more than the available number of cpus.")
    else:
        results = Parallel(n_jobs=cpu_number_one)(
            [
                delayed(SVC_learn)(
                    r,
                    dim,
                    depth,
                    data,
                    labels,
                    CV,
                    regC,
                    reg_gamma,
                    reg_kernel,
                    cpu_number=cpu_number_two,
                )
                for r in scale_factors
            ]
        )
        best = np.argmax(np.delete(results, 1, axis=1), axis=0)[1]
        return results, results[best]


def logistic_learn(
    scale_factor: float,
    dim: int,
    depth: int,
    data: list[np.array],
    labels: list[int],
    CV: Any = None,
    regC: list[int] = [5],
    no_iter: list[int] = [100],
    cpu_number: int = 1,
) -> tuple[float, LogisticRegressionCV, float]:
    """
    Fit a LogisticRegressionCV model to the input data, using the given scale factor.

    The entries in the data list are scaled via the sig_scale_depth_ratio
    function using scale_factor as the scale_factor, i.e via
    sig_scale_depth_ratio(data, dim, depth, scale_factor=scale_factor).

    Once scaled, a sklearn GridSearchCV is run with the model set to be
    LogisticRegressionCV(random_state=42), the param_grid to be {'Cs':regC ,
    'max_iter':no_iter} and the cross-validation strategy to be determined by CV.

    The selected best model is used to predict the labels for the appropriately
    scaled data, and the accuracy_score of the predicted labels compared to the
    actual labels is computed.

    The returned output is a list composed of the scale_factor used, the model
    selected during the GridSearch, and the accuracy_score achieved by the
    selected model.

    Parameters
    ----------
    scale_factor : float
        The scaling factor to be applied to the signature terms
    dim : int
        Dimension of the input stream
    depth : int
        Level to which the input signature is truncated
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]
        containing the signature of the stream
    labels : list[int]
        List of integer labels marking the number corresponding to each sequence
    CV : Any, optional
        Must be an int, cross-validation generator or an iterable.
        See scikit-learn documentation for more details, by default None
    regC : list[int], optional
        List of possible Cs inputs for a LogisticRegressionCV model, by default [5]
    no_iter : list[int], optional
        List of possible maximum number of iterations for a LogisticRegressionCV
    cpu_number : int, optional
        The number of cpus to use for parallelisation, by default 1

    Returns
    -------
    tuple[float, LogisticRegressionCV, float]
        A tuple containing the scale_factor used, the model selected during the
        GridSearch, and the accuracy_score achieved by the selected model
    """
    if depth == 0:
        return print(
            "Error: Depth 0 term of signature is always 1 and will not change under scaling"
        )
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    else:
        LR = LogisticRegressionCV(random_state=42)
        tuned_params = {"Cs": regC, "max_iter": no_iter}
        Q = Parallel(n_jobs=cpu_number)(
            [
                delayed(sig_scale_depth_ratio)(data[k], dim, depth, scale_factor)
                for k in range(len(data))
            ]
        )
        model = GridSearchCV(
            estimator=LR, param_grid=tuned_params, cv=CV, n_jobs=cpu_number
        )
        model.fit(Q, labels)
        best_model = model.best_estimator_
        preds = best_model.predict(Q)
        acc = accuracy_score(preds, labels)
        return scale_factor, best_model, acc


def logistic_scale_learn(
    scale_factors: list[float],
    dim: int,
    depth: int,
    data: list[np.array],
    labels: list[int],
    CV: Any = None,
    regC: list[int] = [5],
    no_iter: list[int] = [100],
    cpu_number_one: int = 1,
    cpu_number_two: int = 1,
) -> tuple[
    list[tuple[float, LogisticRegressionCV, float]],
    tuple[float, LogisticRegressionCV, float],
]:
    """
    Fit a LogisticRegressionCV model to the input data, using the given scale factors.

    The returned outputs are a list of the results of logistic_learn(r, dim,
    depth, data, labels, CV, regC, no_iter) for each entry r in
    scale_factors, and a separate recording of the output corresponding to
    the entry r in scale_factors for which the best accuracy_score is achieved.

    The integer cpu_number_one determines how many cpus are used for
    parallelisation over the scale_factors, whilst cpu_number_two determines the
    number of cpus used for parallelisation during the call of the function
    logistic_learn for each scale_factor considered.

    Parameters
    ----------
    scale_factors : list[float]
        List of scaling factors to be applied to the signature terms
    dim : int
        Dimension of the input stream
    depth : int
        Level to which the input signature is truncated
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]
        containing the signature of the stream
    labels : list[int]
        List of integer labels marking the number corresponding to each sequence
    CV : Any, optional
        Must be an int, cross-validation generator or an iterable.
        See scikit-learn documentation for more details, by default None
    regC : list[int], optional
        List of possible Cs inputs for a LogisticRegressionCV model, by default [5]
    no_iter : list[int], optional
        List of possible maximum number of iterations for a LogisticRegressionCV
    cpu_number_one : int, optional
        The number of cpus to use for parallelisation over the scale_factors,
        by default 1
    cpu_number_two : int, optional
        The number of cpus to use for parallelisation during the call of the
        function ridge_learn for each scale_factor considered, by default 1

    Returns
    -------
    tuple[list[tuple[float, LogisticRegressionCV, float]], tuple[float, LogisticRegressionCV, float]]
        A tuple containing a list of the results of logistic_learn(r, dim, depth,
        data, labels, CV, regC, no_iter) for each entry r in scale_factors, and a separate
        recording of the output corresponding to the entry r in scale_factors for which
        the best accuracy_score is achieved
    """
    if depth == 0:
        return print(
            "Error: Depth 0 term of signature is always 1 and will not change under scaling"
        )
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    if cpu_number_one + cpu_number_two > joblib.cpu_count() and (
        cpu_number_one != 1 or cpu_number_two != 1
    ):
        return print("Error: Tried to allocate more than the available number of cpus.")
    else:
        results = Parallel(n_jobs=cpu_number_one)(
            [
                delayed(logistic_learn)(
                    r,
                    dim,
                    depth,
                    data,
                    labels,
                    CV,
                    regC,
                    no_iter,
                    cpu_number=cpu_number_two,
                )
                for r in scale_factors
            ]
        )
        best = np.argmax(np.delete(results, 1, axis=1), axis=0)[1]
        return results, results[best]


def forest_learn(
    scale_factor: float,
    dim: int,
    depth: int,
    data: list[np.array],
    labels: list[int],
    CV: Any = None,
    reg_est: list[int] = [10, 100],
    reg_feat: list[str | int | float] = ["auto"],
    cpu_number: int = 1,
) -> tuple[float, RFC, float]:
    """
    Fit a RandomForestClassifier model to the input data, using the given scale factor.

    The entries in the data list are scaled via the sig_scale_depth_ratio
    function with scale_factor as the scale_factor, i.e
    sig_scale_depth_ratio(data, dim, depth, scale_factor=scale_factor).

    Once scaled, a sklearn GridSearchCV is run with the model set to be
    RFC(random_state=42), the param_grid to be {'max_features':reg_feat,
    'n_estimators':reg_est} and the cross-validation strategy to be determined
    by CV.

    The selected best model is used to predict the labels for the appropriately
    scaled data, and the accuracy_score of the predicted labels compared to the
    actual labels is computed.

    The returned output is a list composed of the scale_factor used, the model
    selected during the GridSearch, and the accuracy_score achieved by the
    selected model.

    Parameters
    ----------
    scale_factor : float
        The scaling factor to be applied to the signature terms
    dim : int
        Dimension of the input stream
    depth : int
        Level to which the input signature is truncated
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]
        containing the signature of the stream
    labels : list[int]
        List of integer labels marking the number corresponding to each sequence
    CV : Any, optional
        Must be an int, cross-validation generator or an iterable.
        See scikit-learn documentation for more details, by default None
    reg_est : list[int], optional
        List of possible n_estimators inputs for a RandomForestClassifier model,
        by default [10,100]
    reg_feat : list[str  |  int  |  float], optional
        List of max_features strategies for a RandomForestClassifier model,
        by default ["auto"]
    cpu_number : int, optional
        The number of cpus to use for parallelisation, by default 1

    Returns
    -------
    tuple[float, RFC, float]
        A tuple containing the scale_factor used, the model selected during the
        GridSearch, and the accuracy_score achieved by the selected model
    """
    if depth == 0:
        return print(
            "Error: Depth 0 term of signature is always 1 and will not change under scaling"
        )
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    else:
        forest = RFC(random_state=42, n_jobs=cpu_number)
        tuned_params = {"n_estimators": reg_est, "max_features": reg_feat}
        Q = Parallel(n_jobs=cpu_number)(
            [
                delayed(sig_scale_depth_ratio)(data[k], dim, depth, scale_factor)
                for k in range(len(data))
            ]
        )
        model = GridSearchCV(
            estimator=forest, param_grid=tuned_params, cv=CV, n_jobs=cpu_number
        )
        model.fit(Q, labels)
        best_model = model.best_estimator_
        preds = best_model.predict(Q)
        acc = accuracy_score(preds, labels)
        return scale_factor, best_model, acc


def forest_scale_learn(
    scale_factors: list[float],
    dim: int,
    depth: int,
    data: list[np.array],
    labels: list[int],
    CV: Any = None,
    reg_est: list[int] = [10, 100],
    reg_feat: list[str | int | float] = ["auto"],
    cpu_number_one: int = 1,
    cpu_number_two: int = 1,
) -> tuple[list[tuple[float, RFC, float]], tuple[float, RFC, float]]:
    """
    Fit a RandomForestClassifier model to the input data, using the given scale factors.

    The returned outputs are a list of the results of forest_learn(r, dim,
    depth, data, labels, CV, reg_est, reg_feat) for each entry r in
    scale_factors, and a separate recording of the output corresponding to
    the entry r in scale_factors for which the best accuracy_score is achieved.

    The integer cpu_number_one determines how many cpus are used for
    parallelisation over the scale_factors, whilst cpu_number_two determines the
    number of cpus used for parallelisation during the call of the function
    forest_learn for each scale_factor considered.

    Parameters
    ----------
    scale_factors : list[float]
        List of scaling factors to be applied to the signature terms
    dim : int
        Dimension of the input stream
    depth : int
        Level to which the input signature is truncated
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]
        containing the signature of the stream
    labels : list[int]
        List of integer labels marking the number corresponding to each sequence
    CV : Any, optional
        Must be an int, cross-validation generator or an iterable.
        See scikit-learn documentation for more details, by default None
    reg_est : list[int], optional
        List of possible n_estimators inputs for a RandomForestClassifier model,
        by default [10,100]
    reg_feat : list[str  |  int  |  float], optional
        List of max_features strategies for a RandomForestClassifier model,
        by default ["auto"]
    cpu_number_one : int, optional
        The number of cpus to use for parallelisation over the scale_factors,
        by default 1
    cpu_number_two : int, optional
        The number of cpus to use for parallelisation during the call of the
        function ridge_learn for each scale_factor considered, by default 1

    Returns
    -------
    tuple[list[tuple[float, RFC, float]], tuple[float, RFC, float]]
        A tuple containing a list of the results of forest_learn(r, dim, depth,
        data, labels, CV, reg_est, reg_feat) for each entry r in scale_factors,
        and a separate recording of the output corresponding to the entry r in
        scale_factors for which the best accuracy_score is achieved
    """
    if depth == 0:
        return print(
            "Error: Depth 0 term of signature is always 1 and will not change under scaling"
        )
    if dim == 1:
        return print("Error: One-dimensionl signatures are trivial")
    if cpu_number_one + cpu_number_two > joblib.cpu_count() and (
        cpu_number_one != 1 or cpu_number_two != 1
    ):
        return print("Error: Tried to allocate more than the number of available cpus.")
    else:
        results = Parallel(n_jobs=cpu_number_one)(
            [
                delayed(forest_learn)(
                    r,
                    dim,
                    depth,
                    data,
                    labels,
                    CV,
                    reg_est,
                    reg_feat,
                    cpu_number=cpu_number_two,
                )
                for r in scale_factors
            ]
        )
        best = np.argmax(np.delete(results, 1, axis=1), axis=0)[1]
        return results, results[best]


def inv_reset_like_sum(stream: np.array) -> np.array:
    """
    Apply the inverse of the reset_like_sum function to the input stream.

    Parameters
    ----------
    stream : np.array
        Stream of data

    Returns
    -------
    np.array
        Stream of data with the n^th entry in the output given by the
        component-wise sum of the first n entries in stream
    """
    if isinstance(stream, np.ndarray) == True:
        A = np.zeros(stream.shape, dtype=float)
        for k in range(len(A)):
            A[k] = np.sum(stream[: k + 1], axis=0).tolist()
        return A
    else:
        return print("Error: Input is not a numpy array")


def kmeans_fit(a: int, data: list[np.array]) -> KMeans:
    """
    Fit a KMeans cluster model to the input data, using a clusters number of a.

    Parameters
    ----------
    a : int
        Integer in [2, len(data) - 1]
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]

    Returns
    -------
    KMeans
        KMeans cluster model, with n_clusters = a, fitted to data
    """
    if a < 2 or a >= len(data):
        return print(
            f"Error: Number of clusters must be an integer in [{2},{len(data) - 1}]"
        )
    else:
        return KMeans(n_clusters=a, random_state=42).fit(data)


def kmeans_cluster_number(
    k_range: set[int], data: list[np.array], cpu_number: int = 1
) -> tuple[int, float]:
    """
    For each integer j in k_range, a KMeans cluster model looking for j clusters
    is fitted to data.

    The silhouette_score of each trained model is computed.

    The returned output is the model achieving the highest silhouette_score and
    its associated silhouette_score

    Parameters
    ----------
    k_range : set[int]
        Set of integers in [2, len(data) - 1]
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]
    cpu_number : int, optional
        The number of cpus to use for parallelisation, by default 1

    Returns
    -------
    tuple[int, float]
        The model achieving the highest silhouette_score and its associated
        silhouette_score
    """
    if min(k_range) < 2 or max(k_range) >= len(data):
        return print(
            f"Error: k_range must be a set of integers within [{2},{len(data) - 1}]"
        )
    else:
        kmeans_per_k = Parallel(n_jobs=cpu_number)(
            [delayed(kmeans_fit)(a, data) for a in k_range]
        )
        silhouette_scores = [
            silhouette_score(data, model.labels_) for model in kmeans_per_k
        ]
        best_index = np.argmax(silhouette_scores)
        best_k = k_range[best_index]
        best_score = silhouette_scores[best_index]
        return best_k, best_score


def kmeans_percentile_propagate(
    cluster_num: int, percent: float, data: list[np.array], labels: list[int]
) -> tuple[list[np.array], list[int]]:
    """
    A KMeans model with cluster_num clusters is used to fit_transform data.

    A representative centroid element from each cluster is assigned its
    corresponding label from the labels list.

    This label is then propagated to the percent% of the instances in the
    cluster that are closest to the clusters centroid.

    The returned output is a list of these chosen instances and a list of the
    corresponding labels for these instances.

    Parameters
    ----------
    cluster_num : int
        Integer in [2, len(data) - 1]
    percent : float
        Float in [0, 100]
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]
    labels : list[int]
        List of integer labels marking the number corresponding to each sequence

    Returns
    -------
    tuple[list[np.array], list[int]]
        A tuple containing a list of the chosen instances and a list of the
        corresponding labels for these instances
    """
    if percent > 100 or percent < 0:
        return print("Error: percent must be in [0,100]")
    if cluster_num > len(data) - 1 or cluster_num < 2:
        return print(f"Error: cluster_num must be an integer in [{2},{len(data) - 1}]")
    else:
        kmeans = KMeans(n_clusters=cluster_num, random_state=42)
        data_dist = kmeans.fit_transform(data)
        data_cluster_dist = data_dist[np.arange(len(data)), kmeans.labels_]
        for i in range(cluster_num):
            in_cluster = kmeans.labels_ == i
            cluster_dist = data_cluster_dist[in_cluster]
            cutoff_distance = np.percentile(cluster_dist, percent)
            above_cutoff = data_cluster_dist > cutoff_distance
            data_cluster_dist[in_cluster & above_cutoff] = -1
        partially_propagated = data_cluster_dist != -1
        data_pp, labels_pp = [], []
        for z in range(len(partially_propagated)):
            if partially_propagated[z] == True:
                data_pp.append(data[z])
                labels_pp.append(labels[z])
        return data_pp, labels_pp


def model_performance(
    B: tuple[float, Any],
    dim: int,
    depth: int,
    data: list[np.array],
    labels: list[int],
    cpu_number: int,
) -> tuple[float, Any, float, np.array, np.array]:
    """
    Compute model performance using the given model B and the input data.
    Compute the accuracy_score, the confusion matrix and a normalised confusion_matrix.

    The entries in the data list are scaled via the sig_scale_depth_ratio
    function with B[0] as the scale_factor, i.e. sig_scale_depth_ratio(data,
    dim, depth, scale_factor = B[0]).

    The model B[1] is used to make predictions using the rescaled data, and the
    accuracy_score, the confusion matrix and a normalised confusion_matrix
    (focusing only on the incorrect classifications) of these predictions
    compared to the labels are computed.

    The returned outputs are the scale_factor B[0] used, the model B[1] used, the
    accuracy_score achieved, the confusion_matrix and the normalised
    confusion_matrix highlighting the errors.

    Parameters
    ----------
    B : tuple[float, Any]
        A tuple containing a float and a trained classifier
    dim : int
        Dimension of the input stream
    depth : int
        Level to which the input signature is truncated
    data : list[np.array]
        List of numpy arrays of shape [signature_terms]
        containing the signature of the stream
    labels : list[int]
        List of integer labels marking the number corresponding to each sequence
    cpu_number : int
        The number of cpus to use for parallelisation

    Returns
    -------
    tuple[float, Any, float, np.array, np.array]
        A tuple containing the scale_factor used, the model selected during the
        GridSearch, the accuracy_score achieved by the selected model, the
        confusion_matrix and the normalised confusion_matrix highlighting the errors
    """
    scaled_data = Parallel(n_jobs=cpu_number)(
        [delayed(sig_scale_depth_ratio)(a, dim, depth, B[0]) for a in data]
    )
    preds = B[1].predict(scaled_data)
    acc = accuracy_score(preds, labels)

    CM = confusion_matrix(preds, labels)
    row_sums = CM.sum(axis=1, keepdims=True)
    for i in range(len(row_sums)):
        if row_sums[i] == 0:
            row_sums[i] = 1
    CM_norm = CM / row_sums
    np.fill_diagonal(CM_norm, 0)

    return B[0], B[1], acc, CM, CM_norm
