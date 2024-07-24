import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


import scanpy as sc

### Discrete colors

d_color1=["#e6194b","#3cb44b","#ffe119",
"#4363d8","#f58231","#911eb4",
"#46f0f0","#f032e6","#bcf60c",
"#fabebe","#008080","#e6beff",
"#9a6324",
#              "#fffac8",
             "#800000",
"#aaffc3","#808000","#ffd8b1",
"#000075","#808080"
#              ,"#ffffff","#000000"
             
             
            ]

d_color2=[
     "#a6cee3",  # Light Blue
    "#1f78b4",  # Dark Blue
    "#8dd3c7",  # Pale Cyan
    "#b2df8a",  # Light Green

    "#33a02c",  # Dark Green
    
    "#fdbf6f",  # Light Orange
    "#ff7f00",  # Dark Orange
    "#b15928",   # Brown
    "#cab2d6",  # Light Purple
    "#6a3d9a",  # Dark Purple
    "#fccde5",  # Pale Pink
     "#fb8072",  # Salmon Pink
    "#ffed6f"   # Light Yellow
]



d_color3=[
"#e6194b","#3cb44b","#ffe119",
"#4363d8","#f58231","#911eb4",
"#46f0f0","#f032e6","#bcf60c",
"#fabebe","#008080","#e6beff",
"#9a6324",
#              "#fffac8",
             "#800000",
"#aaffc3","#808000","#ffd8b1",
"#000075","#808080",
#              ,"#ffffff","#000000"
    
       "#a6cee3",  # Light Blue
    "#1f78b4",  # Dark Blue
    "#8dd3c7",  # Pale Cyan
    "#b2df8a",  # Light Green

    "#33a02c",  # Dark Green
    
    "#fdbf6f",  # Light Orange
    "#ff7f00",  # Dark Orange
    "#b15928",   # Brown
    "#cab2d6",  # Light Purple
    "#6a3d9a",  # Dark Purple
    "#fccde5",  # Pale Pink
     "#fb8072",  # Salmon Pink
    "#ffed6f"   # Light Yellow


]


d_color4=sns.color_palette(np.hstack((d_color3,sc.pl.palettes.godsnot_102)))

### Continous colors
# https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html#sphx-glr-tutorials-colors-colormap-manipulation-py
# # https://gree2.github.io/python/2015/05/06/python-seaborn-tutorial-choosing-color-palettes
from matplotlib import cm
from matplotlib import colors, colorbar
cmap_own = cm.get_cmap('magma_r', 256)
newcolors = cmap_own(np.linspace(0,0.75 , 256))
Greys = cm.get_cmap('Greys_r', 256)
newcolors[:10, :] = Greys(np.linspace(0.8125, 0.8725, 10))
c_color1 = colors.ListedColormap(newcolors)


from matplotlib import cm
from matplotlib import colors, colorbar
cmap_own = cm.get_cmap('RdBu_r', 256)
newcolors = cmap_own(np.linspace(0.15,0.85 , 256))
c_color2=colors.ListedColormap(newcolors)


from matplotlib import cm
from matplotlib import colors, colorbar
cmap_own = cm.get_cmap('YlGn', 256)
newcolors = cmap_own(np.linspace(0.05,0.8 , 256))
c_color3=colors.ListedColormap(newcolors)



import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
def createCustomCmapFromHex(hex_colors):
    """
    Create a custom colormap from a list of hex colors.

    Parameters:
    hex_colors (list): A list of color codes in hex format.

    Returns:
    LinearSegmentedColormap: The custom colormap.
    """
    # Convert hex colors to RGB
    rgb_colors = [mcolors.hex2color(color) for color in hex_colors]
    # Create a new colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", rgb_colors)
    return cmap

hex_colors = ['#faefef', '#e8aebc', '#d96998', '#b1257a', '#572266']
c_color4 = createCustomCmapFromHex(hex_colors)

