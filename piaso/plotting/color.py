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

    # "#33a02c",  # Dark Green
    
    "#fdbf6f",  # Light Orange
    # "#ff7f00",  # Dark Orange
    "#b15928",   # Brown
    # "#cab2d6",  # Light Purple
    # "#6a3d9a",  # Dark Purple
    # "#fccde5",  # Pale Pink
     "#fb8072",  # Salmon Pink
    "#ffed6f"   # Light Yellow


]


# d_color4=sns.color_palette(np.hstack((d_color3,sc.pl.palettes.godsnot_102)))
d_color4=list(np.hstack((d_color3,sc.pl.palettes.godsnot_102)))


np.random.seed(30303) 
d_color5=list(np.random.choice(d_color3, size=len(d_color3), replace=False,))

np.random.seed(333) 
d_color6=list(np.random.choice(d_color3, size=len(d_color3), replace=False,))

np.random.seed(333222) 
d_color7=list(np.random.choice(d_color3, size=len(d_color3), replace=False,))


#### Generate larger color set
colors_pool = [
    
        ## Blue
        '#BDE4F4', '#93DCFC', '#5B89D8',"#4363d8",
        # '#526FB3', #### not very distinctive
        '#074b90',
     # '#000075',### Too dark
    
        ## Green
       '#bcf60c',  '#D1D32C', '#75F798', '#8AC434', '#BBDE8A', "#3cb44b",
        '#D6DAB9', 
        # '#BDCDB6',### too similar
        # '#AFC9A6', ### too similar
       # '#83B097',### not very distinctive
       '#7DA496' ,'#B2DBCE', '#64ACA8',  '#508876', 
    
    
        ## Purple
        '#949EC4', '#B8AED0',  '#8E8DE1','#911eb4', 
      # '#983680', ### Too similar,
      '#735C94', 
       # '#45345E', ## too dark
    '#f032e6',
    
        
        
        ## Grey
        # '#FFFBEF',## Too white, too dim
       '#E8E5DB', '#ffd8b1', '#808080', 
        
        ## Yellow
        # '#F8F8D3', ### Too bright
        # '#FAFFB8', ### Too bright
        '#FCE38A', "#ffe119", '#FFA06F', 
       # '#FFD700', ### Not too distinctive
        # '#FF7F50', ### Not too distinctive
    '#f58231',
    
        ## Red
        # '#EE3862', ## Not very distinctive
        '#F84914',
         # '#C42536', ## Not very distinctive
         "#e6194b",
        
        ## Pink
       '#F4DBCD', 
       # '#EBCCCD', ### To similar to the other color
       '#F0CFD4', '#FFD9E8', '#E7B1AF', '#F198CC','#DE95BA',  '#B76E79',
    
        ## Brown 
        '#BC938B', '#D59D29', '#9a6324',
    
       
       ]



np.random.seed(5230) 
d_color8=list(np.random.choice(colors_pool, size=36, replace=False,))

np.random.seed(3691) 
d_color9=list(np.random.choice(colors_pool, size=36, replace=False,))

np.random.seed(305230) 
d_color10=list(np.random.choice(colors_pool, size=36, replace=False,))

np.random.seed(20250224) 
d_color11=list(np.random.choice(colors_pool, size=36, replace=False,))



### Reorder indices:
import numpy as np
def _reorder_numbers(n, step):
    """
    Generates an array from 0 to n-1 and reorders it in a step-wise column-first pattern.

    Parameters
    ----------
    n : int
        The total number of elements to generate.
    step : int
        The step size to group elements.

    Returns
    -------
    numpy array
        Reordered array following the pattern 0, step, 2*step, ..., 1, step+1, ...
    
    Example
    -------
    >>> reorder_numbers(42, 5)
    array([ 0,  5, 10, 15, 20, 25, 30, 35, 40,  
            1,  6, 11, 16, 21, 26, 31, 36, 41,  
            2,  7, 12, 17, 22, 27, 32, 37,  
            3,  8, 13, 18, 23, 28, 33, 38,  
            4,  9, 14, 19, 24, 29, 34, 39])
    """
    row_size = step
    col_size = -(-n // step)  # Equivalent to ceil(n / step)

    # Create the array from 0 to n-1
    arr = np.arange(n)

    # Pad array to fit into a perfect grid
    padded_arr = np.pad(arr, (0, row_size * col_size - n), mode='constant', constant_values=-1)

    # Reshape, transpose, and flatten
    reordered = padded_arr.reshape(col_size, row_size).T.flatten()

    # Remove padding values
    return reordered[reordered != -1]

new_indices=_reorder_numbers(len(colors_pool), 4)
d_color12=[colors_pool[i] for i in new_indices]
new_indices=_reorder_numbers(len(colors_pool), 3)
d_color13=[colors_pool[i] for i in new_indices]


colors_pool2=np.hstack((colors_pool, d_color4))
np.random.seed(333) 
d_color14=list(np.random.choice(colors_pool2, size=len(colors_pool2), replace=False,))

np.random.seed(1830) 
d_color15=list(np.random.choice(colors_pool2, size=len(colors_pool2), replace=False,))


## celeste
d_color16 = ['#1a2c6cff', '#8369c2ff', '#7f7f53ff', '#ccae94ff', '#b0d8feff', '#bcfe7eff', '#845d4dff', '#bebf9fff',
           '#173c4eff', '#d3adfeff', '#a1197dff', '#69c8fbff', '#ffff80ff', '#ed78b6ff', '#ac6581ff', '#8a457bff',
           '#ce8e87ff', '#d0338fff', '#fe8a8aff', '#6a42d7ff', '#351ebeff', '#b02121ff', '#78f9feff', '#5ef54fff',
           '#0e62c0ff', '#009819ff', '#57eedbff', '#c4f888ff','#fcd6cfff', '#5b16b9ff', '#b22cefff', '#cd7bc3ff',
           '#d7985aff']

## coco
d_color17 = ['#110453ff', '#822283ff', '#bf7cd1ff', '#465dadff', '#c79093ff', '#70023bff', '#f27371ff', '#a1d1e7ff',
        '#ef3f22ff', '#fba01dff', '#f5811fff', '#ef509dff', '#2597c7ff', '#057ebaff', '#53a28aff', '#3c2587ff',
        '#a4a227ff', '#6d4869ff', '#edd767ff', '#bd0e15ff', '#174816ff', '#bbcc83ff', '#a0565eff', '#b0713fff',
        '#c86581ff', '#67daedff', '#5d626bff', '#f2c420ff', '#b6e3b1ff', '#8c5373ff', '#ad97e1ff', '#ef7eacff']

## stray
d_color18 = ['#900896ff', '#b354b6ff', '#b3de70ff', '#8ae883ff', '#8e8e2cff', '#31dcc6ff', '#64f7ffff', '#f14ef6ff',
         '#f1ff30ff', '#cc8d08ff', '#8ca681ff', '#11701cff', '#34ac73ff', '#c51d42ff', '#b04607ff', '#f66e01ff',
         '#33f167ff', '#1d5dc9ff', '#bff328ff', '#1a1259ff', '#4892a6ff', '#a2dbc9ff', '#b1b3caff', '#154f1dff',
         '#ffc14eff', '#2e2f9eff', '#653892ff', '#90634cff', '#5e2525ff', '#af2616ff', '#4e508dff', '#733036ff']

## goose
d_color19 = ['#3d6bd5ff', '#ed2027ff', '#5f94d0ff', '#f7c7baff', '#d8d3bcff', '#cfbd7fff', '#29518eff', '#7d5c53ff',
         '#44403fff', '#d2a419ff', '#89b55aff', '#af4c66ff', '#6d8188ff', '#6d4467ff', '#7c694eff', '#cc816dff',
         '#a7cfe8ff', '#415141ff', '#7fb18eff', '#4f6d4fff', '#ff9727ff', '#abaaaaff', '#e1a775ff', '#ac76c4ff',
         '#fee298ff', '#f3dc59ff', '#af7b25ff', '#e39192ff', '#5bb6a1ff', '#aa549aff', '#7859b6ff', '#c45141ff',
         '#cdd6abff', '#7c3849ff', '#e05846ff', '#519148ff', '#4b5785ff', '#c5ad37ff', '#b9562cff', '#700000ff']

## fall_guys
d_color20 = ['#ffd567ff', '#fc9eecff', '#d17cfeff', '#84f8f3ff', '#33cb64ff', '#ef919aff', '#ffe3c6ff', '#f93b71ff',
             '#5abdf8ff', '#9344e3ff', '#f1bceaff', '#c5eaf6ff', '#b23385ff', '#ca603bff', '#356dc7ff', '#1b48bdff',
             '#fefea0ff', '#0d9034ff', '#ff8e26ff', '#ffef1dff', '#fd3301ff', '#f968b7ff', '#62059cff', '#7ed9a3ff',
             '#f9aa7dff', '#3b9f9aff', '#b2e683ff', '#f030dbff', '#abc032ff', '#5cfdacff', '#29d0f3ff', '#03bd82ff',
             '#8268a3ff', '#181a7fff', '#c21e3aff', '#7a7890ff', '#d9d5f7ff', '#f25f26ff', '#c6bc89ff', '#94a3f5ff']



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
    This function converts a sequence of hex colors into an RGB-based colormap that can be used for visualizations in Matplotlib.

    Parameters
    ----------
    hex_colors : list of str
        A list of color codes in hexadecimal format (e.g., `['#faefef', '#e8aebc', '#d96998', '#b1257a', '#572266']`).


    Returns
    -------
    LinearSegmentedColormap
        A Matplotlib `LinearSegmentedColormap` object that can be applied to plots using the `cmap` parameter.
        
    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import piaso
    >>> # Define custom hex colors
    >>> hex_colors = ['#faefef', '#e8aebc', '#d96998', '#b1257a', '#572266']
    >>> 
    >>> # Create the colormap
    >>> c_color4 = piaso.pl.color.createCustomCmapFromHex(hex_colors)
    >>> 
    >>> # Generate a gradient to visualize the colormap
    >>> gradient = np.linspace(0, 1, 256).reshape(1, -1)
    >>> 
    >>> # Display the colormap
    >>> plt.figure(figsize=(6, 1))
    >>> plt.imshow(gradient, aspect="auto", cmap=c_color4)
    >>> plt.axis("off")
    >>> plt.show()
    """
    # Convert hex colors to RGB
    rgb_colors = [mcolors.hex2color(color) for color in hex_colors]
    # Create a new colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", rgb_colors)
    return cmap

### Create a cmap
hex_colors = ['#faefef', '#e8aebc', '#d96998', '#b1257a', '#572266']
c_color4 = createCustomCmapFromHex(hex_colors)

### Dark blues
c_color5=createCustomCmapFromHex(['#EFEEEC', '#B2C5E4', '#547BB9', '#1E3569'])
c_color5

### Lake greens
c_color6=createCustomCmapFromHex(['#EFEEEC', '#DFE6E1', '#B2D9D8', '#87C3D2', '#3B7E92', '#4B7280'])
c_color6

### Reds
c_color7=createCustomCmapFromHex([ '#EFEEEC',  '#FFA06F', '#F58231', '#F84914', "#e6194b",])
c_color7

### Purples
c_color8=createCustomCmapFromHex(['#EFEEEC',  '#B8AED0',  '#735C94', '#45345E',])
c_color8


