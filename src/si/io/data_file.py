import numpy as np

from ..data.dataset import Dataset


def read_data_file(
    filename,
    sep=",",
    label=True
):


    data = np.genfromtxt(
        filename,
        delimiter=sep
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if label:
        X = data[:, :-1]
        y = data[:, -1]
        label_name = "y"
    else:
        X = data
        y = None
        label_name = None

    feature_names = [
        f"feat_{index}"
        for index in range(X.shape[1])
    ]

    return Dataset(
        X=X,
        y=y,
        features=feature_names,
        label=label_name
    )


def write_data_file(
    filename,
    dataset,
    sep=",",
    label=True
):
    """
    Writes a Dataset object to a data file.
    """

    if label and dataset.y is not None:
        data = np.column_stack(
            (dataset.X, dataset.y)
        )
    else:
        data = dataset.X

    np.savetxt(
        filename,
        data,
        delimiter=sep
    )