import seaborn as sns
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
LINEWIDTH = 3
FIG_DPI = 100
sns.set(font_scale=3, style="white", rc={
    "lines.linewidth": 3,
    "lines.markersize":20,
    "ps.useafm": True,
    "font.sans-serif": ["Helvetica"],
    "pdf.use14corefonts" : True,
    "text.usetex": False,
    })
filetype = "png"
mpl.rcParams['hatch.linewidth'] = 2.0

def draw_grouped_barplot_three_subbars(data_df, colors, x_label: str, y_label: str, seaborn_hue: str, 
    bargraph_savepath: str, title: str="", y_axis_name:str="", ylim_top: float=None, amazon_data_flag=False, 
    bbox_to_anchor=(0, -0.35,1,0.2), position=None, figsize=(11, 4)):      

    print("Saving the plot in ", bargraph_savepath)    
    fig = plt.figure(figsize=figsize)
    if position is not None:
        ax = fig.add_axes(position)
    else:
        ax = fig.add_subplot(111)
    data_df = data_df.sort_values(by=[x_label])
    if not amazon_data_flag:        
        new_data_df = data_df.loc[data_df[x_label] == 'Yelp']
        for val in ['IMDB', 'Tripadvisor']:
            new_data_df = new_data_df.append(data_df.loc[data_df[x_label] == val])        
        data_df = new_data_df
    else:
        new_data_df = data_df.replace("Cellphones", "Cell")
        new_data_df = new_data_df.replace("Luxury Beauty", "Beauty")
        new_data_df = new_data_df.replace("Automotive", "Auto")
        new_data_df = new_data_df.replace("Pet Supplies", "Pet")
        data_df = new_data_df

    patterns = ['', '//', '-']    

    subx = data_df[seaborn_hue].unique()    
    u = data_df[x_label].unique()
    x = np.arange(len(u))
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width = np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = data_df[data_df[seaborn_hue] == gr]
        ax.bar(x+offsets[i], dfg[y_label].values, width=width,
                hatch=patterns[i], 
                color=colors[i])
    plt.xticks(x, u)
    ax = plt.gca()
    
    ax.set_ylim(40, 100)
    ax.get_yaxis().get_offset_text().set_x(-0.05)
    ax.set_xlabel('')
    ax.set_ylabel(y_axis_name)

    ax.tick_params(axis='x', which='major')
    ax.tick_params(axis='y', which='major')
    
    legend_texts = ["all sentences", "sentences with\nnegation", "sentences with\npositive lexicon"]
    lg = plt.legend(legend_texts, loc='upper center', mode='expand', bbox_to_anchor=bbox_to_anchor, frameon=False)
    plt.savefig(bargraph_savepath+"."+filetype, 
            dpi=FIG_DPI)