import pandas as pd
from nltk import regexp_tokenize
from collections import defaultdict, namedtuple, Counter
import numpy as np
import matplotlib.pyplot as plt

"""
Split the data into ONLY NECESSARY COLUMNS per GROUP (NT; MH; HS; UV)

"""
def open_file(filename):
    df = pd.read_csv(filename, delimiter=',')
    return df
  

def filter_columns(df):
    """
    Only the new classifications for each group in defaultdict
    """
    
    df = df[["AnsökanID",'DiarieförtÅr', "ÖnskadeBeredningsgrupperKortNamn",'TilldeladBeredningsgruppKortNamn', "AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "BeskrivningEng", "InternaForskningsämnenSCBKoder", "InternaForskningsämnenSCB", "Nyckelord"]]

    #all_groups = defaultdict(lambda:defaultdict(int))
    div_groups = defaultdict(set)
    for _, row in df.iterrows():
        bg = row['TilldeladBeredningsgruppKortNamn']
        #print(bg)
        if '-' in bg and bg.split('-')[0] != 'U':
            #print(bg)
            group = bg.split('-')[0]
            subgroup = bg.split('-')[1]

            if group == 'NT' and subgroup.isdigit():
                continue
            if group == 'MH' and not subgroup[0].isdigit():
                continue

            #if len(subgroup) > 2:
            #    subgroup = subgroup[:2]
            
            
            div_groups[group].add(bg)

            #if it its the same group
            #all_groups[group][subgroup] += 1 
            #all_groups[group]['Total'] += 1
    
    #print(all_groups)
    return div_groups        
        
def dataset_save(div_groups, df):
    
    for group, new_groups in div_groups.items():
                
        f_df = df[df['TilldeladBeredningsgruppKortNamn'].isin(new_groups)].reset_index(drop=True)
        f_df.to_csv(f'./datasets/{group}/{group}_dataset.csv', index=False)
    


if __name__ == "__main__":
    df = open_file("vr_data/DATA_copy.csv")
    div_groups=filter_columns(df)
    dataset_save(div_groups, df)
    