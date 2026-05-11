"""
This module deals with creating a tokeniser model and tokenising input text.
"""
import sentencepiece as spm
from utils.path_manager import PathManager as pm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SPTokenizer:
    """
    This class handles the tokenisation for the CNN and the RNN. It utilises SentencePiece with BPE-encoding
    to assume a similar approach to the RoBERTa tokeniser.

    Attributes:
        label (str) = The label column name to be used for comparing model predictions against.
        columns (list(str)) = The columns to be used for information to predict classes.
        label2id (dict) = A dictionary converting labels to integers.
        id2label (dict) = A dictionary converting integers to labels.

    Methods:
        label_extractor(self, df: pd.DataFrame) -> None 
        checker(self, model: str) -> True|False
        create_vocab(self, df: pd.DataFrame, model: str) -> None
        get_vocab(self, model: str) -> spm.SentencePieceProcessor
        tokenizer(self, df: pd.DataFrame)

    Returns:
        checker -> True | False
        get_vocab() -> sp (SentencePiece model)
        tokenizer() -> t_tokens, t_labels (Tensorised tokens and labels)
    """

    def __init__(self, text_columns: list[str], label: str, df: pd.DataFrame=None, model: str='tokenizer'):

        self.pm = pm("./")

        self.label = label
        self.columns = text_columns 
        self.label2id = {}
        self.id2label = {}
        if self.checker(model) is False:
            self.create_vocab(df, model)
            print("The tokenizer is created")
        else:
            self.model = self.get_vocab(model)

    def label_extractor(self, df: pd.DataFrame) -> None:
        """
        Creates a mapping for label to integer and integer to dictionary in two separate
        dictionaries.

        Arguments:
            df (pd.DataFrame) = The dataframe from which to extract the labels.
        """
        for i, label in enumerate(sorted(df[self.label].unique())):
            self.label2id[label] = i
            self.id2label[i] = label

    def checker(self, model: str) -> True|False:
        """
        Checks whether a tokeniser model has already been created.

        Arguments:
            model(str) = The name of the tokeniser model.

        Returns:
            True | False, depending on whether the tokeniser model already exists or not.
        """
        if not self.pm.get_tok(model):
            print(self.pm.get_tok(model))
            return False
        return True
    
    def create_vocab(self, df: pd.DataFrame, model: str) -> None:
        """
        Creates the vocabulary for the tokeniser model on the full dataset.

        Arguments:
            df(pd.DataFrame) = The dataframe containing all of the data and all classes for maximum vocabulary.
            model(str) = The name of the tokeniser model.

        Returns:
            None
        """
        print(df) 

        #text_columns = ["AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "Nyckelord"]
        print("This is in create vocab", self.columns)
        all_text = pd.concat([df[col].dropna() for col in self.columns], ignore_index=True)

        self.pm.setup_tok()

        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(all_text.astype(str)), #input tp construct vocab
            model_prefix = str(self.pm.tokenizer_dir / model),
            vocab_size = 16000,
            character_coverage=1.0, #all unique characters
            model_type = 'bpe', 
            minloglevel=2
        )
        
    
    def get_vocab(self, model: str) -> spm.SentencePieceProcessor:
        """
        Retrieves the previously trained tokeniser model.

        Arguments:
            model(str) = The name of the tokeniser model.

        Returns:
            sp (SentencePiece object) = The trained sentencepiece tokeniser.
        """
        #load the tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(str(self.pm.tokenizer_dir/f"{model}.model"))
        
        return sp

    
    def tokenizer(self, df: pd.DataFrame):
        """
        This method deals with tokenising the input text.

        Arguments:
            df (pd.DataFrame) = The dataframe from which to retrieve text and tokenise.
        
        Returns:
            t_tokens (torch.Tensor) = A tensorised version of the tokens.
            t_labels (torch.Tensor) = A tensorised version of the labels.
        """
        import inspect
        #print("TOKENIZER CALLED FROM:", inspect.getfile(inspect.currentframe()))

        tokens = []
        labels = []

        for _, row in df.iterrows():

            all_text = " ".join([str(row[col]) for col in self.columns if pd.notna(row[col])])
            label = row[self.label]
            


            token_ids = self.model.encode(all_text, add_bos=True, add_eos=True, out_type=int)
            label_idx = self.label2id[label]

            tokens.append(torch.tensor(token_ids))
            labels.append(torch.tensor(label_idx))
        #print(f"label2id keys sample: {list(self.label2id.keys())}")
        t_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
        #t_tokens = torch.stack(tokens).unsqueeze(1)
        t_labels = torch.stack(labels)#.unsqueeze(1)

        return t_tokens, t_labels
