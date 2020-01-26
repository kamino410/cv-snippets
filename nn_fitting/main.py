import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data_x = np.arange(0, 10, 1e-2).reshape([-1, 1])
data_y = np.exp(data_x/10)*np.cos(data_x)

mlp = make_pipeline(StandardScaler(),
                    MLPRegressor(verbose=True, hidden_layer_sizes=(30, 30, 30),
                                 max_iter=1500, random_state=0, activation='logistic'))
mlp.fit(data_x, data_y)


import plotly.offline as po
import plotly.graph_objs as go

est_y = mlp.predict(data_x)

trace1 = go.Scatter(x=data_x.flatten(), y=data_y.flatten(), mode='markers')
trace2 = go.Scatter(x=data_x.flatten(), y=est_y.flatten(), mode='markers')
data = [trace1, trace2]

layout = dict(title='test')

fig = go.Figure(data=data, layout=layout)
po.plot(fig, filename='test.html')
