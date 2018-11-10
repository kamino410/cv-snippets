import plotly.offline as po
import plotly.graph_objs as go
import numpy as np

# plotly.offline.init_notebook_mode(connected=True)

data = [go.Mesh3d(
    x=[0, 5, 5, 0],
    y=[1, 1, 1, 1],
    z=[0, 0, 5, 5],
    i=[0, 0],
    j=[1, 2],
    k=[2, 3],
    opacity=0.4
)]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
po.plot(fig, filename='sample-plane.html')
