import pandas as pd

def filter_on_overperform(df,group,measure,factor=2):

    grouped = df.groupby(group)[measure].mean()
    grouped_dict = {group:val for group,val in df.groupby(group)[measure].\
        mean().iteritems()}
    keep_index = set([i for i,r in df.iterrows()\
        if r[measure] >= grouped_dict[r[group]]*factor])
    new_df = df[df.index.isin(keep_index)]

    return new_df
