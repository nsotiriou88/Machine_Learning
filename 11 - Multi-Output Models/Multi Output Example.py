# Simple Multiple Output Example

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

X = np.random.random((10,3))
y = np.random.random((10,2))
X2 = np.random.random((7,3))

knn = KNeighborsRegressor()
regr = MultiOutputRegressor(knn)

regr.fit(X,y)
regr.predict(X2)
