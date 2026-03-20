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

#All the training , param_hunt, eval -> experiment handler
#fix the columns and maybe pm

#TODO Organiser in exp_handler
#TODO Main

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-bg', type=str, default='NT')
    parser.add_argument('--create_datasets', '-cd', action='store_true', default=False) 
    parser.add_argument('-m', type=str, default='roberta') #model
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

    if args.file is None and args.create_datasets:
        print('Please specify path to dataset')
        break
     
    if args.columns is None and args.bg:
        df = pd.read_csv(args.f).columns.values
        df_dict = {i: v for i,v in enumerate(df)}
        columns = input(f"{df_dict} \nKindly select numbers of columns to be used for data:")
        args.columns = [df_dict[col] for column in columns]

    if args.label is None:
        df = pd.read_csv(args.f).columns.values

    config = Config(
        model = Model(args.m),
        k = args.k,
        batch_size = args.batch_size,
        n_epochs = args.e,
        lr = args.lr,
        dropout = args.dr,
        columns=args.columns,
        label=args.label)
    
    #train(config)
    



if __name__ == "__main__":
    main()




