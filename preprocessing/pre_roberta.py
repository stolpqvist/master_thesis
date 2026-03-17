"""
This module is the main module that handles the preprocessing required for the RoBERTa model.
It takes a filename when creating the class, then handles id creation and preprocessing.
It outputs a stacked tensor.

Class:
    DataProcessor: handles the data preprocessing of the roberta model.

Attributes:
    file(str): the filename/path to dataset.
    label2id(defaultdict(int)): the label to id assigner that defines the id as the length of the dictionary.
    tokenizer (AutoTokenizer) = the tokenizer chosen for the XLMRobertaForSequenceClassification model.

Returns:
    row_yielder = yields a row in a dataframe.
    preprocessing = returns a list of tokenized torch.Tensors.

"""
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
import torch

COLUMN_BUDGETS = {
    "AnsökanTitel":         50,
    "AnsökanTitelEng":      50,
    "Sammanfattning":       190,
    "Populärbeskrivning":   190,
    "Nyckelord":            32,
}





class DataProcessor(Dataset):
    """
    This class handles dataprocessing of the RoBERTa model.

    Attributes:
        file, str = the path to a particular file
        label2id, defaultdict(int) = the labels and their corresponding id's
        id2label, dict = the ids and their corresponding labels
        tokenizer, AutoTokenizer = the tokenizer chosen for the XLMRobertaForSequenceClassification model

    Methods:
        row_yielder = Opens and reads chosen dataframe and yields row by row
        preprocessing = Processes the dataframe,
    """
    def __init__(self, df):
        self.df = df
        self.label2id = {} #defaultdict(lambda: len(self.label2id))
        self.id2label = {}
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.text_columns = ["AnsökanTitel", "AnsökanTitelEng", "Sammanfattning", "Populärbeskrivning", "Nyckelord"] 
        self.label_column = 'TilldeladBeredningsgruppKortNamn'


    def label_extractor(self) -> None:
        """
        Extracts the labels and enters it into a defauldict to handle label -> id and
        id -> label assignment.
        """

        #add to labels
        for i, label in enumerate(np.unique(self.df[self.label_column].values)):
            self.label2id[label] = i
            self.id2label[i] = label

        self._pretokenize()
        del self.df #delete after tokenizing

    def _pretokenize(self):
        print("Pre-tokenizing..")

        self.samples = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            all_input_ids = [torch.tensor([self.tokenizer.cls_token_id])]
            all_attention_masks = [torch.tensor([1])]

            for col, budget in COLUMN_BUDGETS.items():
                tok = self.tokenizer(
                    str(row[col]),
                    max_length=budget,
                    truncation = True,
                    padding = "max_length",
                    add_special_tokens = False,
                    return_tensors="pt"
                )

                all_input_ids.append(tok["input_ids"].squeeze(0))
                all_attention_masks.append(tok["attention_mask"].squeeze(0))
        
            all_input_ids.append(torch.tensor([self.tokenizer.sep_token_id]))
            all_attention_masks.append(torch.tensor([1]))

            label = self.label2id[row[self.label_column]]

            self.samples.append((
                {
                    "input_ids": torch.cat(all_input_ids),
                    "attention_mask": torch.cat(all_attention_masks)
                },
                torch.tensor(label, dtype=torch.long)
            ))

        print(f"Done. {len(self.samples)} samples ready")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
        #row = self.df.iloc[idx]
        
        #all_input_ids = [torch.tensor([self.tokenizer.cls_token_id])]
        #all_attention_masks = [torch.tensor([1])]
        

        #for col, budget in COLUMN_BUDGETS.items():
        #    tok = self.tokenizer(
        #        str(row[col]),
        #       max_length=budget,
        #        truncation = True,
        #        padding = "max_length",
        #        add_special_tokens = False,
        #        return_tensors="pt"
        #    )

        #    all_input_ids.append(tok["input_ids"].squeeze(0))
        #    all_attention_masks.append(tok["attention_mask"].squeeze(0))
        
        #all_input_ids.append(torch.tensor([self.tokenizer.sep_token_id]))
        #all_attention_masks.append(torch.tensor([1]))

        #label = self.label2id[row[self.label_column]]

        #return {
        #    "input_ids": torch.cat(all_input_ids),
        #    "attention_mask": torch.cat(all_attention_masks)
        #}, torch.tensor(label, dtype=torch.long)

    
     



    
