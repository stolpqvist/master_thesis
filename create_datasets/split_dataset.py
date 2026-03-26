import pandas as pd
#from nltk import regexp_tokenize
from collections import defaultdict
import numpy as np
#import matplotlib.pyplot as plt
from utils.path_manager import PathManager

"""
Split the data into ONLY NECESSARY COLUMNS per GROUP (NT; MH; HS; UV)

"""
class GroupSplit:
    def __init__(self, pm: PathManager, columns, label, df): #filepath="../vr_data/DATA_copy.csv"):
        self.pm = pm
        #self.filename = filepath
        self.label = label
        self.columns = columns
        self.df = df
        self.groups = self.filter_columns(self.df, self.columns, self.label)
        self.dataset_save(self.groups, self.df, self.label)



    #def open_file(self):
    #    df = pd.read_csv(self.filename, delimiter=',')
    #    return df
    

    def filter_columns(self, df, columns, label):
        """
        Only the new classifications for each group in defaultdict
        """
        
        df = df[columns]#["AnsökanID",'DiarieförtÅr', "ÖnskadeBeredningsgrupperKortNamn",'TilldeladBeredningsgruppKortNamn', "AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "BeskrivningEng", "InternaForskningsämnenSCBKoder", "InternaForskningsämnenSCB", "Nyckelord"]]

        div_groups = defaultdict(set)
        for _, row in df.iterrows():
            bg = row[label] #'TilldeladBeredningsgruppKortNamn']
            #print(bg)
            if '-' in bg and bg.split('-')[0] != 'U':
                #print(bg)
                group = bg.split('-')[0]
                subgroup = bg.split('-')[1]

                if group == 'NT' and subgroup.isdigit():
                    continue
                if group == 'MH' and not subgroup[0].isdigit():
                    continue
                
                div_groups[group].add(bg)

        
        #print(all_groups)
        return div_groups        
            
    def dataset_save(self, div_groups, df, label):
        
        for group, new_groups in div_groups.items():

            self.pm.get_group_dir(group) #GROUP dirs are created

            #f_df = df[df['TilldeladBeredningsgruppKortNamn'].isin(new_groups)].reset_index(drop=True)
            f_df = df[df[label].isin(new_groups)].reset_index(drop=True)

            f_df.to_csv(self.pm.get_dataset_csv(group), index=False) #SAVING the group csv for each group
    


#if __name__ == "__main__":
#    df = open_file("vr_data/DATA_copy.csv")
#    div_groups=filter_columns(df)
#    dataset_save(div_groups, df)
    