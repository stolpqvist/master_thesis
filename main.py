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

    parser.add_argument('-bg', type=str, default=None, help='The group the model should train on.')
    parser.add_argument('--create_datasets', '-cd', action='store_true', default=False, help='Creates datasets for the models to be trained on, not available during thesis submission.') 
    parser.add_argument('-m',"--model",nargs='+', type=str, default='roberta', help='Which model to train, only the flag --boot accepts multiple models during selection') #model
    parser.add_argument('-dr', type=float, default=0.1, help='The dropout rate to be applied to the model.')
    parser.add_argument('-lr', type=float, default=0.00001, help='The learning rate to be applied to the model.')
    parser.add_argument('-e', type=int, default=10, help='Controls how many epochs to train over.') #epochs
    parser.add_argument('--batch_size', '-bs', type=int, default=2, help='The amount of batches to load the data in.')
    parser.add_argument('-k', type=int, default=10, help='The amount of K-folds to create.')
    parser.add_argument('-tr', '--train', action='store_true', default=False, help='Enables training.')
    parser.add_argument('--param_hunt', '-p', action='store_true', default=False, help='Enables hyperparameter optimisation.')
    parser.add_argument('-test_size', type=float, default=0.1, help='During creation of datasets, controls test set size.')  
    parser.add_argument('--test', action='store_true', default=False, help='Enables testing.')     # Test size
    parser.add_argument('-c','--columns', default=None, help='The data columns to be used for training model.')
    parser.add_argument('-l', '--label', default=None, help='The label column to be used for testing predictions.')
    parser.add_argument('-f', '--file', default=None, help='Controls which file to read from if -bg is not present.')
    parser.add_argument('-b', '--boot', action='store_true', default=False, help='Enables bootstrapping.') 
    parser.add_argument('-v', '--vis', action='store_true', default=False, help='Enables visualisation.')
    parser.add_argument('-em','--emissions', action='store_true', default=False, help='Enables emissions tracking.')

    args = parser.parse_args()
    if args.create_datasets and args.file is None:
        raise Exception("To create datasets a full dataset has to be provided,\
                    \nin order for it to be split.")
    

    file = args.file or args.bg

    if args.bg:
        pm = PathManager()


        for model in args.model:
            pm.setup_result(model)


        if args.train or args.param_hunt:  #MOVE BOOT TO TEST_SET

            pm.setup_boot() 

            file = pm.get_trainval_csv(args.bg)

        if args.test or args.boot:
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




