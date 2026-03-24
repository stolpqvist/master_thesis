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
#fix the columns and maybe pm

#TODO Organiser in exp_handler
#TODO Main

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-bg', type=str, default='NT')
    parser.add_argument('--create_datasets', '-cd', action='store_true', default=False) 
    parser.add_argument('-m',"--model", type=str, default='roberta') #model
    parser.add_argument('-dr', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=0.00001)
    parser.add_argument('-e', type=int, default=5) #epochs
    parser.add_argument('--batch_size', '-b', type=int, default=3)
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-tr', action='store_true', default=False)
    parser.add_argument('--param_hunt', '-p', action='store_true', default=False)
    parser.add_argument('-test_size', type=float, default=0.1)       # Test size
    parser.add_argument('-c','--columns', default=None, help='The data columns to be used for training model')
    parser.add_argument('-l', '--label', default=None, help='The label column to be used for testing predictions')
    parser.add_argument('-f', '--file', default=None) 


    args = parser.parse_args()

    if args.file is None:
        if args.create_datasets:
            raise Exception("To create datasets a full dataset has to be provided,\
                    \nin order for it to be split.")
        else:
            raise Exception("Kindly provide a file to be used as a dataset.")

    if args.columns is None and args.bg:
        df = pd.read_csv(args.file).columns.values
        df_dict = {i: v for i,v in enumerate(df)}
        columns = input(f"{df_dict} \nKindly select numbers of columns to be used for data: ")
        args.columns = [df_dict[int(column)] for column in columns]
        print(f"The columns chosen are: {args.columns}\n")

    if args.label is None:
        df = pd.read_csv(args.file).columns.values
        df_dict = {i:v for i,v in enumerate(df)}
        label = input(f"{df_dict} \nKindly select the label column to be used for classification: ")
        args.label = [df_dict[int(l)] for l in label]
        if len(args.label) > 1:
            raise Exception("Only one label column can be chosen")

    config = Config(
        model = Model(args.model),
        k = args.k,
        batch_size = args.batch_size,
        n_epochs = args.e,
        lr = args.lr,
        dropout = args.dr,
        columns=args.columns,
        label=args.label)

    exp = ExperimentOrganiser(
            model_name =    args.model,
            bg =            args.bg,
            columns =       args.columns,
            label =         args.label,
            lr =            args.lr,
            dropout =       args.dr,
            epochs =        args.e,
            batch_size =    args.batch_size
            )
    
    #train(config)
    



if __name__ == "__main__":
    main()




