import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = np.loadtxt('./c2p_dmd.csv', delimiter=',')

data_x = data[:, 0:2]
data_y = data[:, 2:4]

mlp = make_pipeline(StandardScaler(),
                    MLPRegressor(verbose=True, hidden_layer_sizes=(100, 100, 100),
                                 max_iter=1500, random_state=0, activation='logistic'))
mlp.fit(data_x, data_y)


import plotly.offline as po
import plotly.graph_objs as go

est_y = mlp.predict(data_x[::500])

trace1 = go.Scatter(
    x=data_y[::500, 0], y=data_y[::500, 1], mode='markers', marker=dict(size=2))
trace2 = go.Scatter(x=est_y[:, 0], y=est_y[:, 1],
                    mode='markers', marker=dict(size=2))
data = [trace1, trace2]

layout = dict(title='test')

fig = go.Figure(data=data, layout=layout)
po.plot(fig, filename='test.html')
