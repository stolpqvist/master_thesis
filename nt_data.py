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
    
    #TOKENIZER

    #pattern = r'''(?x)
    #(?:[A-Z]\.)+ #Abbrev
    #| \w+(?:-\w+)* #words with hyhens
    #| \€?\d+(?:\.\d+)?%? # currency
    #| \.\.\. # ellipsis
    #| [][,;"'?():_'-] # these are separate tokens; includes ], [
#'''
    #for i, value in enumerate(df):
    #    if i >= 10:
    #        break
    #    tokenised_text = regexp_tokenize(value, pattern)
        #print(tokenised_text)

def filter_columns(df):
    #f_1 = df.head(1)
    #print(first_row)
    #df.drop['AnsökanStatus', 'AntalÖnskadeBeredningsgrupper', 'SöknyckelText', 'Stödform', 'BidragsformKortnamn', 'Inriktning']
    
    df = df[['DiarieförtÅr', 'TilldeladBeredningsgruppKortNamn','InternaForskningsämnenSCBKoder', 'InternaForskningsämnenSCB']]
    #print(df.columns.tolist())
    #print(df.shape, df.head())
    #print(type(df))

    all_groups = defaultdict(list)

    scb_code_name = namedtuple('scb_code_name', ['code'])
    for _, row in df.iterrows():
        bg = row['TilldeladBeredningsgruppKortNamn']
        #print(bg)
        if bg.split('-')[0].startswith('NT'):
            #print(bg)
            codes = row['InternaForskningsämnenSCBKoder']
            #words = row['InternaForskningsämnenSCB'] 

            if ';' in codes:
                code = codes.split(';')[0]
                #word = words.split(';')[0]
            else:
                code = codes
                #word = words
            
                
            if bg in all_groups:
                #add the codes
                values = {number.code for number in all_groups[bg]}
                #print(values)

                if code not in values:
                    all_groups[bg].append(scb_code_name(code=code))
            else:
                all_groups[bg].append(scb_code_name(code=code))

    #print(all_groups)
    return all_groups

def old_new_groups(all_groups):
    """
    extract 2 groups from all BG (2015-20 /21-25)
    """
    old_groups = defaultdict(list)
    new_groups = defaultdict(list)

    for group in all_groups.keys():
        group_name = group.split('-')[1].strip()

        if group_name.isdigit():
            value = all_groups[group]
            old_groups[group].extend(value)
        else:
            value = all_groups[group]
            new_groups[group].extend(value)

    return old_groups, new_groups
    #print('old', old_groups.keys(), 'new', new_groups.keys())

def match_groups(old_groups, new_groups):
    """
    match the old groups to new ones based on the SCB codes
    """
    group_map = {}
    #print(old_groups)
    for og in old_groups:
        old_codes = {number.code for number in old_groups[og]}
        #print(codes)
        best_n = 0
        for ng in new_groups.keys():
            new_codes = {element.code for element in new_groups[ng]}
            #print(codes)
            n = len(old_codes & new_codes)
            if n > best_n:
                best_n = n
                group_map[og] = ng
    #print(group_map)
    return group_map  

#ccheck the appl from ONE OLD GR -> based on SCB code where it classifies in the new group 
# OLD / NEW (count of gr)          
def classification(df, new_groups):
    """
    classify the app-s from OLD based on all numbers to a specific gr from NEW
    """
    df = df[['DiarieförtÅr', 'TilldeladBeredningsgruppKortNamn','InternaForskningsämnenSCBKoder']]
    
    group_move_counter = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        bg = row['TilldeladBeredningsgruppKortNamn']
        #print(bg)
        if '-' in bg:
            group, number = bg.split('-')
            #print(group, number)
            if group == 'NT' and number.isdigit():
                #print("Banana bread")
                codes = row['InternaForskningsämnenSCBKoder']

                if ';' in codes:
                    if len(codes.split(';')) > 2:
                        all_codes = set(codes.split(';')) 

                    else: 
                        all_codes = set(codes.split(';'))

                else:
                    all_codes = set(codes.strip().split(';'))
                new_codes = set()
                for code in all_codes:
                    ccode = code.strip()
                    new_codes.add(ccode)
                bn_matches = 0
                best_group = defaultdict()

                for ng in new_groups:
                    #print(ng)
                    values = {group.code for group in new_groups[ng]}

                    #print(values)
                    #print("These are the application codes: ",new_codes)
                    #print("These are the beredningsgrupp specific SCB codes: ",values)
                    n_matches = len(new_codes & values)
                    matches = new_codes & values

                    #print("These are the matches: ",matches)

                    if n_matches > bn_matches:
                        #best_group = ng
                        bn_matches = n_matches
                        best_group[ng] = n_matches
                        
            #sort groups from highest to lowest
                #print(best_group)
                best_group = defaultdict(int, sorted(best_group.items(), key=lambda item: item[1], reverse=True))
                #print(best_group, "after sorting")
                #print(bg)
                #print(new_codes, values, ng)
                if len(list(best_group.keys())) > 0:
                    
                    ng = next(iter(best_group))
                    group_move_counter[bg][ng] += 1
    #print(group_move_counter)
    return group_move_counter


def get_visual(group_move_counter):
    """
    make a graph of our mapping distribtion
    """

    for og, inner_dict in group_move_counter.items():
        
        sorted_items = sorted(inner_dict.items(), key=lambda x: x[1], reverse=True)
        new_groups = list(sorted_items)
        print(new_groups) #<- tuple


        #values = [inner_dict[k] for k, v in new_groups]


        top_values = [v for k, v in new_groups[:4]]
        other_sum = sum(int(v) for k, v in new_groups[4:])
        values = top_values + [other_sum]

        new_groups = [k for k, v in new_groups[:4]] + ['Other']


        plt.figure()

        plt.bar(new_groups, values)
        plt.title(f" Reassigned from {og}")
        plt.xlabel("New Distribution")
        plt.ylabel("Number")
        plt.savefig(f"Images/New groups for {og}.png", dpi=300, bbox_inches='tight')
        plt.show()
    


if __name__ == "__main__":
    df = open_file("../vr_data/DATA_copy.csv")
    all_groups=filter_columns(df)
    old, new = old_new_groups(all_groups)

    #for group in old:
    #    print(old[group])
    #match_groups(old, new)
    new_res=classification(df, new)
    get_visual(new_res)


#WHAT DO WE KNOW:
#FOR CLASSIFICATION -> codes are not the best
#For features -> Title, Abstract, (ENG/SWED) , Key words + maybe codes