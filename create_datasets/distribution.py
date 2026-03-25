import pandas as pd
from nltk import regexp_tokenize
from collections import defaultdict, namedtuple, Counter
import numpy as np
import matplotlib.pyplot as plt

"""
Opening the csv
Creating a dict with keys as Beredningsgrupper and values as named_tuple (code, key word)
TO DO:
1. check if 2 and 3 codes will improve the matching -> they do for each appl classification

"""
def open_file(filename):
    df = pd.read_csv(filename, delimiter=',')#['AnsökanTitel']

    return df
  

def filter_columns(df):
    """
    Only the new classifications for each group in defaultdict
    """
    
    df = df[['DiarieförtÅr', 'TilldeladBeredningsgruppKortNamn']]

    all_groups = defaultdict(lambda:defaultdict(int))

    for _, row in df.iterrows():
        bg = row['TilldeladBeredningsgruppKortNamn']
        #print(bg)
        if '-' in bg and bg.split('-')[0] != 'U':
            #print(bg)
            group = bg.split('-')[0]
            subgroup = bg.split('-')[1]

            if group == 'UV' and not subgroup.isdigit():
                    continue
            if group == 'NT' and subgroup.isdigit():
                continue
            if group == 'MH' and not subgroup[0].isdigit():
                continue

            if len(subgroup) > 2:
                subgroup = subgroup[:2]
            
            #if it its the same group
            all_groups[group][subgroup] += 1 
            all_groups[group]['Total'] += 1
    
    #print(all_groups)
    return all_groups

         

def get_visual(all_groups):
    """
    make a graph of each group distribtion
    """

    for og, inner_dict in all_groups.items():
        
        sorted_items = sorted(inner_dict.items(), key=lambda x: x[0], reverse=False)
        new_groups = list(sorted_items)
        #print(new_groups) #<- tuple

        total_g, total_v = new_groups[-1]

        values = [v for k, v in new_groups[:-1]]
        new_groups = [k for k, v in new_groups[:-1]]



        plt.figure()

        plt.bar(new_groups, values)
        plt.title(f" {og} Total: {total_v}")
        plt.xlabel("New Distribution")
        plt.ylabel("Number")
        plt.savefig(f"../Images/Groups/Distribution for {og}.png", dpi=300, bbox_inches='tight')
        plt.show()
    


if __name__ == "__main__":
    df = open_file("../vr_data/data_uppdaterad.csv")
    all_groups=filter_columns(df)
    get_visual(all_groups)