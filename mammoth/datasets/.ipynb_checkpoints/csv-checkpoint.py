import numpy as np
from mammoth.datasets.dataset import Dataset
from sklearn import preprocessing

def _features(df, numeric, categorical,transform=False):
    import pandas as pd

    if transform:
        numeric_parts = [pd.DataFrame(_transform(df[col], "numerical")) for col in numeric]
        cat_parts = [pd.DataFrame(_transform(df[col], "categorical")) for col in categorical]
        dfs = numeric_parts + cat_parts
    else:
        dfs = [df[col] for col in numeric] + [
            pd.get_dummies(df[col]) for col in categorical
        ]
    return pd.concat(dfs, axis=1).values

def _transform(col,col_type="categorical"):
    if col_type=="categorical":
        col=col.fillna("missing")
        lb = preprocessing.LabelBinarizer()
        col=lb.fit_transform(col)
    else:
        col = col.fillna(0)
        arr_2d = col.values.reshape(-1, 1)
        scaler = preprocessing.StandardScaler()
        col= scaler.fit_transform(arr_2d)
    return col



class CSV(Dataset):
    def __init__(self, data, numeric, categorical, labels, sensitives=None):
        import pandas as pd

        self.data = data
        self.numeric = numeric
        self.categorical = categorical
        self.labels = (
            pd.get_dummies(data[labels])
            if isinstance(labels, str)
            else (labels if isinstance(labels, dict) else {"label": labels})
        )
        self.cols = numeric + categorical
        if sensitives is None:
            sensitives = []
        self.pred_cols = [col for col in self.cols if col not in sensitives]

    def to_features(self, sensitive):
        return _features(self.data, self.numeric, self.categorical).astype(np.float64)

    def to_pred(self, sensitive):
        num=[col for col in self.numeric if col not in sensitive]
        cat=[col for col in self.categorical if col not in sensitive]
        return _features(self.data, num, cat,transform=True).astype(np.float64)
