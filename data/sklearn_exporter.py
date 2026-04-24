#fmt: off

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

# prepare data
df = pd.read_csv("bank.csv", sep=";")
df = df.dropna()
X = df.drop("y", axis=1)
y = (df["y"] == "yes").astype(int)
numeric_features = ["age","balance","day","duration","campaign","pdays","previous",]
categorical_features = ["job","marital","education","default","housing","loan","contact","month","poutcome",]

# train
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
model = LogisticRegression(max_iter=200)
classifier = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
classifier.fit(X, y)  # in the future: classifier.predict(X)

initial_types = [(col, FloatTensorType([None, 1])) for col in numeric_features]+[(col, StringTensorType([None, 1])) for col in categorical_features]
onnx_model = convert_sklearn(classifier, initial_types=initial_types)
with open("bank_model.onnx", "wb") as f: f.write(onnx_model.SerializeToString())



import numpy as np
import fairbench as fb
from pygrank.algorithms.autotune.optimization import optimize

base_model = classifier
marital_column = df["marital"]
marital_column_values = df["marital"].unique()

def build_classifier(params):
    classifier = base_model
    print(params)
    a = marital_column.map(dict(zip(marital_column_values,params[:len(marital_column_values)])))
    b = marital_column.map(dict(zip(marital_column_values,params[len(marital_column_values):])))
    for _ in range(4):
        yhat = classifier.predict_proba(X)[:,1]
        yhat = (yhat-yhat.min())/(yhat.max()-yhat.min())
        err = (yhat-y)**2
        weights = a*np.exp(b*err)+(1-a)*np.exp(-b*err)
        weights = weights/weights.max()
        model = LogisticRegression(max_iter=200)
        classifier = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        classifier.fit(X, y, model__sample_weight=weights)
    return classifier

sensitive = fb.Dimensions(fb.categories@df["marital"])
def assess(params):
    classifier = build_classifier(params)
    yhat = classifier.predict(X)
    acc = fb.measures.acc(yhat,y)
    dfpr = fb.quick.maxdiff_tpr_pairwise(predictions=yhat,labels=y,sensitive=sensitive)
    dfnr = fb.quick.maxdiff_tnr_pairwise(predictions=yhat,labels=y,sensitive=sensitive)
    relpr = fb.quick.maxrel_pr_pairwise(predictions=yhat,labels=y,sensitive=sensitive)
    return -2*float(acc)+float(relpr)+float(dfpr)+float(dfnr)+2


params = optimize(assess, max_vals=[1,1,1,3,3,3], min_vals=[0,0,0,-3,-3,-3], verbose=True, randomize=True)
print("final params: "+str(params))
classifier = build_classifier(params)
fb.reports.pairwise(predictions=base_model.predict(X), labels=y, sensitive=sensitive).show(env=fb.export.ConsoleTable)
fb.reports.pairwise(predictions=classifier.predict(X), labels=y, sensitive=sensitive).show(env=fb.export.ConsoleTable)
onnx_model = convert_sklearn(classifier, initial_types=initial_types)
with open("bank_model_weighted.onnx", "wb") as f: f.write(onnx_model.SerializeToString())