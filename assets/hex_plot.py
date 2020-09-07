#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:23:15 2020

@author: kevincory
"""


import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import matplotlib.cm as cm
import cmocean#  http://matplotlib.org/cmocean/

import plotly.graph_objs as go


#points = df_sub[['LOC_X', 'LOC_Y']].copy()



def get_hexbin_attributes(hexbin):
    paths = hexbin.get_paths()
    points_codes = list(paths[0].iter_segments()) #path[0].iter_segments() is a generator 
    prototypical_hexagon = [item[0] for item in points_codes]
    return prototypical_hexagon, hexbin.get_offsets(), hexbin.get_facecolors(), hexbin.get_array()

def pl_cell_color(mpl_facecolors):
     
    return [ f'rgb({int(R*255)}, {int(G*255)}, {int(B*255)})' for (R, G, B, A) in mpl_facecolors]

def make_hexagon(prototypical_hex, offset, fillcolor, linecolor=None):
   
    new_hex_vertices = [vertex + offset for vertex in prototypical_hex]
    vertices = np.asarray(new_hex_vertices[:-1])
    # hexagon center
    center=np.mean(vertices, axis=0)
    if linecolor is None:
        linecolor = fillcolor
    #define the SVG-type path:    
    path = 'M '
    for vert in new_hex_vertices:
        path +=  f'{vert[0]}, {vert[1]} L' 
    return  dict(type='path',
                 line=dict(color=linecolor, 
                           width=0.5),
                 path=  path[:-2],
                 fillcolor=fillcolor, 
                ), center

def mpl_to_plotly(cmap, N):
    h = 1.0/(N-1)
    pl_colorscale = []
    for k in range(N):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([round(k*h,2), f'rgb({C[0]}, {C[1]}, {C[2]})'])
    return pl_colorscale


def hexbin_graph(HB):
    
    """
    df_sub = df2[df2['PLAYER_NAME'] == player].copy()
    print(df_sub.shape)
    #plt.figure(figsize=(0.05,0.05))
    #plt.axis('off')
    HB = plt.hexbin(x = 'LOC_X', y = 'LOC_Y', data = df_sub, gridsize= 25, cmap=cmocean.cm.algae , mincnt=1) # cmocean.cm.algae is a cmocean colormap
    print(len(HB.get_edgecolors()))
    """
    
    hexagon_vertices, offsets, mpl_facecolors, counts = get_hexbin_attributes(HB)
    hexagon_vertices[:-1]# the last vertex coincides with the first one
    cell_color = pl_cell_color(mpl_facecolors)
    shapes = []
    centers = []
    print(len(offsets), len(cell_color), mpl_facecolors, cell_color)
    for k in range(len(offsets)):
        print(k)
        shape, center = make_hexagon(hexagon_vertices, offsets[k], cell_color[k])
        shapes.append(shape)
        centers.append(center)
        
    
    pl_algae = mpl_to_plotly(cmocean.cm.algae, 11)
    pl_algae
    
    
    X, Y = zip(*centers)
    
    #define  text to be  displayed on hovering the mouse over the cells
    text = [f'x: {round(X[k],2)}<br>y: {round(Y[k],2)}<br>counts: {int(counts[k])}' for k in range(len(X))]
    
    trace = go.Scatter(
                 x=list(X), 
                 y=list(Y), 
                 mode='markers',
                 marker=dict(size=0.5, 
                             color=counts, 
                             colorscale=pl_algae, 
                             showscale=True,
                             colorbar=dict(
                                         thickness=20,  
                                         ticklen=4
                                         )),             
               text=text, 
               hoverinfo='text'
              )
    
    axis = dict(showgrid=False,
               showline=False,
               zeroline=False,
               ticklen=4 
               )
    
    layout = go.Layout(title='Hexbin plot',
                       width=530, height=550,
                       xaxis=axis,
                       yaxis=axis,
                       hovermode='closest',
                       shapes=shapes,
                       plot_bgcolor='black')
    
    fig = go.FigureWidget(data=[trace], layout=layout)
    return(fig)


"""
def create_hb(player, df2):
    df_sub = df2[df2['PLAYER_NAME'] == player].copy()
    print(df_sub.shape)
    #plt.figure(figsize=(0.05,0.05))
    plt.axis('off')
    HB = plt.hexbin(x = 'LOC_X', y = 'LOC_Y', data = df_sub, gridsize= 25, cmap=cmocean.cm.algae , mincnt=1) # cmocean.cm.algae is a cmocean colormap
    #print(len(HB.get_edgecolors()))
    return(HB)

"""












