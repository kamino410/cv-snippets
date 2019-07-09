import numpy as np

import plotly.offline as po
import plotly.graph_objs as go

xs = np.linspace(0, 100, 101)
ys1 = np.random.randn(len(xs))
ys2 = np.random.randn(len(xs))

trace1 = go.Scatter(x=xs, y=ys1, mode='markers')
trace2 = go.Scatter(x=xs, y=ys2, mode='lines+markers')

data = [trace1, trace2]
layout = dict(title='template of scatter')

fig = go.Figure(data=data, layout=layout)
po.plot(fig, filename='template_scatter.html')
