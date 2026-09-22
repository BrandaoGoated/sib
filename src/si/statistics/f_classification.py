from scipy.stats import f_oneway


def f_classification(dataset):
    """
    Calculate ANOVA F-values and p-values for every feature.

    Parameters
    ----------
    dataset : Dataset
        Labeled Dataset used for classification.

    Returns
    -------
    tuple
        Tuple containing the F-values and p-values.
    """
    if not dataset.has_label():
        raise ValueError(
            "f_classification requires a Dataset with labels."
        )

    classes = dataset.get_classes()

    groups = [
        dataset.X[dataset.y == class_label]
        for class_label in classes
    ]

    f_values, p_values = f_oneway(*groups)

    return f_values, p_values