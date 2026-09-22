import numpy as np

from ..data.dataset import Dataset


def train_test_split(
    dataset,
    test_size=0.2,
    random_state=None
):
    
    if test_size <= 0 or test_size >= 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = dataset.X.shape[0]

    indices = np.random.permutation(n_samples)

    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    if dataset.has_label():
        train_y = dataset.y[train_indices]
        test_y = dataset.y[test_indices]
    else:
        train_y = None
        test_y = None

    train_dataset = Dataset(
        X=dataset.X[train_indices],
        y=train_y,
        features=dataset.features,
        label=dataset.label
    )

    test_dataset = Dataset(
        X=dataset.X[test_indices],
        y=test_y,
        features=dataset.features,
        label=dataset.label
    )

    return train_dataset, test_dataset

def stratified_train_test_split(
    dataset,
    test_size=0.2,
    random_state=None
):
    """
    Split a Dataset into stratified train and test Datasets.

    """
    if not dataset.has_label():
        raise ValueError(
            "Stratified split requires a Dataset with labels."
        )

    if test_size <= 0 or test_size >= 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if random_state is not None:
        np.random.seed(random_state)

    train_indices = []
    test_indices = []

    for class_label in dataset.get_classes():
        class_indices = np.where(
            dataset.y == class_label
        )[0]

        shuffled_indices = np.random.permutation(
            class_indices
        )

        n_test = int(len(class_indices) * test_size)

        test_indices.extend(
            shuffled_indices[:n_test]
        )

        train_indices.extend(
            shuffled_indices[n_test:]
        )

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    train_dataset = Dataset(
        X=dataset.X[train_indices],
        y=dataset.y[train_indices],
        features=dataset.features,
        label=dataset.label
    )

    test_dataset = Dataset(
        X=dataset.X[test_indices],
        y=dataset.y[test_indices],
        features=dataset.features,
        label=dataset.label
    )

    return train_dataset, test_dataset