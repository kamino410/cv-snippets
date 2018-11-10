import plotly.offline as po
import plotly.graph_objs as go
import numpy as np

# plotly.offline.init_notebook_mode(connected=True)

import plotly.figure_factory as ff

fig = ff.create_trisurf(
    x=[0, 1, 2, 0],
    y=[0, 0, 1, 2],
    z=[0, 2, 0, 1],
    simplices=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
po.plot(fig, filename="sample-mesh.html")
