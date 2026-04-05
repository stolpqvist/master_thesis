import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import  train_test_split
from data_handling.strat_fold import StratifiedFold
from config import Config, Model

from preprocessing.pre_roberta import DataProcessor
from model.roberta import CustomXLMRoberta
from train.train import ModelTrain
import numpy as np
from scipy import stats
import copy
from utils.path_manager import PathManager
from create_datasets.split_dataset import GroupSplit
from time import sleep
from experiment_handler import ExperimentOrganiser
#All the training , param_hunt, eval -> experiment handler

def check_col(file):
        df = pd.read_csv(file).columns.values
        df_dict = {i: v for i,v in enumerate(df)}
        columns = input(f"{df_dict}")
        all_columns = [df_dict[int(column)] for column in columns.split()]
        print(f"You chose: {all_columns}\n")
        return all_columns

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-bg', type=str, default=None)
    parser.add_argument('--create_datasets', '-cd', action='store_true', default=False) 
    parser.add_argument('-m',"--model",nargs='+', type=str, default='roberta') #model
    parser.add_argument('-dr', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=0.00001)
    parser.add_argument('-e', type=int, default=5) #epochs
    parser.add_argument('--batch_size', '-bs', type=int, default=3)
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-tr', '--train', action='store_true', default=False)
    parser.add_argument('--param_hunt', '-p', action='store_true', default=False)
    parser.add_argument('-test_size', type=float, default=0.1)  
    parser.add_argument('--test', action='store_true', default=False)     # Test size
    parser.add_argument('-c','--columns', default=None, help='The data columns to be used for training model')
    parser.add_argument('-l', '--label', default=None, help='The label column to be used for testing predictions')
    parser.add_argument('-f', '--file', default=None)
    parser.add_argument('-b', '--boot', action='store_true', default=False) 
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-em','--emissions', action='store_true', default=False)

    args = parser.parse_args()

    if args.create_datasets and args.file is None:
        raise Exception("To create datasets a full dataset has to be provided,\
                    \nin order for it to be split.")
    

    file = args.file or args.bg

    if args.bg:
        pm = PathManager()


        for model in args.model:
            pm.setup_result(model)


        if args.train or args.param_hunt or args.boot: #MOVE BOOT TO TEST_SET

            pm.setup_boot() 

            file = pm.get_trainval_csv(args.bg)

        if args.test:
            pm.setup_boot()
            file = pm.get_test_csv(args.bg)


    if args.columns is None:
        print("\nKindly select numbers of columns to be used for data (space-separated)")
        args.columns = check_col(file)
       

    if args.label is None:
        print("\nKindly select the label column to be used for classification")
        args.label = check_col(file)
        if len(args.label) > 1:
            raise Exception("Only one label column can be chosen")
        print(f"The label chosen is: {args.label}")

    df = pd.read_csv(file)    
        
    config = Config.from_args(args)
    exp = ExperimentOrganiser(df=df, config=config)
    


if __name__ == "__main__":
    main()




