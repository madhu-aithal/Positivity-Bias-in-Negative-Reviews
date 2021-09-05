import seaborn as sns
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
FIG_DPI = 100
sns.set(font_scale=3, style="white", rc={
    "lines.linewidth": 3,
    "lines.markersize":20,
    "ps.useafm": True,
    "font.sans-serif": ["Helvetica"],
    "pdf.use14corefonts" : True,
    "text.usetex": False,
    })
mpl.rcParams['hatch.linewidth'] = 2.0
FILETYPE = ".png"
mpl.rcParams['hatch.color'] = "black"
err_elinewidth = 3

def filter_plot_data(plot_data): 
    amazon_names = ['Pet Supplies', 'Luxury Beauty', 'Automotive', 'Cellphones', 'Sports']
    amazon_names = [val.lower() for val in amazon_names]   
    plot_data_amz = []
    plot_data_non_amz = []
    for d in plot_data:
        if d['name'].lower() in amazon_names:
            plot_data_amz.append(d)
        else:
            plot_data_non_amz.append(d)
    return plot_data_amz, plot_data_non_amz

def shorten_amazon_names(data_df):
    new_data_df = data_df.replace("Cellphones", "Cell")
    new_data_df = new_data_df.replace("Luxury Beauty", "Beauty")
    new_data_df = new_data_df.replace("Automotive", "Auto")
    new_data_df = new_data_df.replace("Pet Supplies", "Pet")
    return new_data_df

def shorten_non_amazon_names(data_df, x_label):
    new_data_df = data_df.loc[data_df[x_label] == 'Yelp']
    for val in ['IMDB', 'SST', 'Tripadvisor', 'PeerRead']:
        new_data_df = new_data_df.append(data_df.loc[data_df[x_label] == val])        
    new_data_df = new_data_df.replace("PeerRead", "Peer")
    new_data_df = new_data_df.replace("Tripadvisor", "TA")
    return new_data_df

def draw_grouped_bargraph_two_subbars(data, x_label, y_label, hue_attribute, 
    plot_savepath, title="", y_axis_name="#tokens", ylim_top=None, 
    amazon_data_flag=False, bbox_to_anchor=None, position=None, figsize=(11, 4), 
    colors = [(84/255,141/255,255/255)]*4):
    print("\n\n")
    print("Saving the plot in ", plot_savepath)
    fig = plt.figure(figsize=figsize)
    if position is not None:
        ax = fig.add_axes(position)
    else:
        ax = fig.add_subplot(1)
    
    data_df = pd.DataFrame({
        x_label: [d[x_label] for d in data],
        y_label: [d[y_label] for d in data],
        hue_attribute: [d[hue_attribute] for d in data],
        "sem_value": [d["sem_value"] for d in data]        
    })

    data_df = data_df.sort_values(by=[x_label, hue_attribute], ascending=[True, False])    
    if not amazon_data_flag:
        data_df = shorten_non_amazon_names(data_df, x_label)
    else:
        data_df = shorten_amazon_names(data_df)
        
    data_df = data_df.loc[data_df['category']!='neutral']
    
    subx = data_df[hue_attribute].unique()    
    u = data_df[x_label].unique()
    x = np.arange(len(u))        
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    
    bar_x_values = {}
    start_values = []
    end_values = []
    all_values = []
    for i,gr in enumerate(subx):        
        dfg = data_df[data_df[hue_attribute] == gr]
        ax.bar(x+offsets[i], dfg[y_label].values, width=width,
                yerr=dfg["sem_value"].values, 
                error_kw={"elinewidth": err_elinewidth},
                color=colors[i])
    plt.xticks(x, u)
    ax = plt.gca()

    if ylim_top != None:
        ax.set_ylim(0, ylim_top)
    ax.get_yaxis().get_offset_text().set_x(-0.05)
    ax.set_xlabel('')
    ax.set_ylabel(y_axis_name)

    for idx, thisbar in enumerate(ax.patches):
        if idx>=len(ax.patches)/2:
            thisbar.set_hatch('/')
            thisbar.set_edgecolor('black')
        else:
            thisbar.set_hatch('o')      
            thisbar.set_edgecolor('black')

    pos_rev_patch = mpl.patches.Patch(facecolor=colors[0], alpha=0.9, edgecolor='black', hatch='o', label='Positive review')
    neg_rev_patch = mpl.patches.Patch(facecolor=colors[1], alpha=0.9, edgecolor='black', hatch='/', label='Negative review')
    lg=plt.legend(handles=[pos_rev_patch, neg_rev_patch], loc='upper left', bbox_to_anchor=bbox_to_anchor, frameon=False)    

    ax.tick_params(axis='x', which='major')
    ax.tick_params(axis='y', which='major')
    
    plt.savefig(plot_savepath+FILETYPE, bbox_extra_artists=(lg,), dpi=FIG_DPI)

def draw_grouped_bargraph_four_subbars(data, x_label, y_label, 
        hue_attribute_1, hue_attribute_2, plot_savepath, 
        title="", ylim_top = None, negation = False,
        y_axis_name="#tokens", amazon_data_flag=False,
        bbox_to_anchor=None, position=None, figsize=(11, 4), liwc_cats=None,
        colors = [(114/255, 200/255, 117/255),(209/255, 68/255, 68/255)]*2,
        review_category_key='review category', 
        legend = ['Negation before \npositive lexicon', 'Negation before \nnegative lexicon'],
        sentiment_category_key = 'text sentiment'):
    print("\n\n")
    print("Saving the plot in ", plot_savepath)    
    
    fig = plt.figure(figsize=figsize)
    if position is not None:
        ax = fig.add_axes(position)
    else:
        ax = fig.add_subplot(111)

    data_df = pd.DataFrame({
        x_label: [d[x_label] for d in data],
        y_label: [d[y_label] for d in data],
        "sem_value": [d["sem_value"] for d in data],
        hue_attribute_1: [d[hue_attribute_1] for d in data],
        hue_attribute_2: [d[hue_attribute_2] for d in data],
        "hue_attribute": [d[hue_attribute_1]+" reviews - "+d[hue_attribute_2] for d in data],
    })
    data_df = data_df.sort_values(by=[x_label, hue_attribute_1, hue_attribute_2], ascending=[True, False, False])    

    if not amazon_data_flag:
        data_df = shorten_non_amazon_names(data_df, x_label)
    else:
        data_df = shorten_amazon_names(data_df)
        
    if liwc_cats is not None:
        data_df = data_df.loc[data_df[review_category_key]!='neutral']        
    else:
        data_df = data_df.loc[data_df[review_category_key]!='neutral']
    
    subx = data_df["hue_attribute"].unique()
    u = data_df[x_label].unique()
    x = np.arange(len(u))
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    
    for i,gr in enumerate(subx):
        dfg = data_df[data_df["hue_attribute"] == gr]
        ax.bar(x+offsets[i], dfg[y_label].values, width=width,
                yerr=dfg["sem_value"].values, 
                error_kw={"elinewidth": err_elinewidth},                
                color=colors[i])
    
    plt.xticks(x, u)    
    ax = plt.gca()
    if ylim_top != None:
        ax.set_ylim(0, ylim_top)

    ax.get_yaxis().get_offset_text().set_x(-0.05)
    ax.set_xlabel('')
    ax.set_ylabel(y_axis_name)
    for idx, thisbar in enumerate(ax.patches):
        if idx>=len(ax.patches)/2:
            thisbar.set_hatch('/')      
            thisbar.set_edgecolor('black')
        else:
            thisbar.set_hatch('o')      
            thisbar.set_edgecolor('black')
        
    if negation:
        pos_sent_patch = mpl.patches.Patch(color=colors[0], label=legend[0])
        neg_sent_patch = mpl.patches.Patch(color=colors[1], label=legend[1])
    else:
        if liwc_cats is not None:
            pos_sent_patch = mpl.patches.Patch(color=colors[0], label=liwc_cats[0])
            neg_sent_patch = mpl.patches.Patch(color=colors[1], label=liwc_cats[1])
        else:
            pos_sent_patch = mpl.patches.Patch(color=colors[0], label='Positive lexicon')
            neg_sent_patch = mpl.patches.Patch(color=colors[1], label='Negative lexicon')

    pos_rev_patch = mpl.patches.Patch(facecolor=(1,1,1), alpha=1, edgecolor='black', hatch='o', label='Positive review')
    neg_rev_patch = mpl.patches.Patch(facecolor=(1,1,1), alpha=1, edgecolor='black', hatch='/', label='Negative review')

    sentiment_bbox_to_anchor = list(bbox_to_anchor)
    sentiment_bbox_to_anchor[-1] -= 0.4
    
    sentiment_legend = plt.legend(handles=[pos_sent_patch, neg_sent_patch], bbox_to_anchor=sentiment_bbox_to_anchor, frameon=False)
    plt.gca().add_artist(sentiment_legend)
    lg=plt.legend(handles=[pos_rev_patch, neg_rev_patch], bbox_to_anchor=bbox_to_anchor, frameon=False)    

    ax.tick_params(axis='x', which='major')
    ax.tick_params(axis='y', which='major')

    plt.savefig(plot_savepath+FILETYPE, dpi=FIG_DPI)