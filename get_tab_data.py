def get_dataset_table(dataset, target_col):
    if target_col not in dataset.columns:
        raise ValueError(
            f"'{target_col}' no existe. Columnas: {dataset.columns.tolist()}"
        )
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]
    return X, y
