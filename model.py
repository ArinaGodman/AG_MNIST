import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import joblib

mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')

X, y = mnist["data"], mnist["target"]

std_scaler = StandardScaler()

X = std_scaler.fit_transform(X)

joblib.dump(std_scaler, 'std_scaler.sav')

svc_best = SVC(random_state=42, C=0.5, gamma=1, kernel='poly')

svc_best.fit(X, y)

joblib.dump(svc_best, 'svc_model.sav')

print('Klart!')