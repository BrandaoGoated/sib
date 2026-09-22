import pandas as pd

from ..data.dataset import Dataset


def read_csv(
    filename,
    sep=",",
    features=True,
    label=True
):

    if features:
        data = pd.read_csv(
            filename,
            sep=sep,
            header=0
        )
    else:
        data = pd.read_csv(
            filename,
            sep=sep,
            header=None
        )

        data.columns = [
            f"feat_{index}"
            for index in range(data.shape[1])
        ]

    if label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()

        feature_names = data.columns[:-1].tolist()
        label_name = data.columns[-1]
    else:
        X = data.to_numpy()
        y = None

        feature_names = data.columns.tolist()
        label_name = None

    return Dataset(
        X=X,
        y=y,
        features=feature_names,
        label=label_name
    )


def write_csv(
    filename,
    dataset,
    sep=",",
    features=True,
    label=True
):

    data = dataset.to_dataframe()

    if not label and dataset.y is not None:
        data = data.iloc[:, :-1]

    data.to_csv(
        filename,
        sep=sep,
        header=features,
        index=False
    )