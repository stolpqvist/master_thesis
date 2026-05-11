"""
This module is the main module that handles the preprocessing required for the XLM-RoBERTa model.
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
    "Sammanfattning":       189,
    "Populärbeskrivning":   189,
    "Nyckelord":            32,
}





class DataProcessor(Dataset):
    """
    This class handles dataprocessing of the RoBERTa model.

    Attributes:
        df (pd.DataFrame) = the path to a particular file.
        label2id (dict) = the labels and their corresponding id's.
        id2label (dict) = the ids and their corresponding labels.
        tokeniser (AutoTokenizer) = the tokenizer chosen for the XLMRobertaForSequenceClassification model.
        text_columns (list(str)) = The columns of information to predict classes on.
        label_column (str) = The label header to be used for comparing model comparisons against.

    Methods:
        label_extractor() Extracts the labels and converts them into integer:label pairs and label:integer pairs.
        pretokenise() Tokenises the text.
        __len__() Returns the length of samples.
        __getitem__(idx(int)) Retrieves a specific item at a specific index.
    """
    def __init__(self, df: pd.DataFrame, columns: list[str], label: str) -> None:
        self.df = df
        self.label2id = {} #defaultdict(lambda: len(self.label2id))
        self.id2label = {}
        self.tokeniser = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.text_columns = columns #["AnsökanTitel", "AnsökanTitelEng", "Sammanfattning", "Populärbeskrivning", "Nyckelord"] 
        self.label_column = label #'TilldeladBeredningsgruppKortNamn'


    def label_extractor(self) -> None:
        """
        Extracts the labels and converts them to integer:label and label:integer pairs.

        Returns:
            None
        """

        #add to labels
        for i, label in enumerate(sorted(self.df[self.label_column].unique())):
            self.label2id[label] = i
            self.id2label[i] = label

        self.pretokenise()
        del self.df #delete after tokenising

    def pretokenise(self) -> None:
        """
        Tokenises the dataframe and columns provided at class init.
        """
        print("Pre-tokenising..")

        self.samples = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            all_input_ids = [torch.tensor([self.tokeniser.cls_token_id])]
            all_attention_masks = [torch.tensor([1])]

            for col, budget in COLUMN_BUDGETS.items():
                tok = self.tokeniser(
                    str(row[col]),
                    max_length=budget,
                    truncation = True,
                    padding = "max_length",
                    add_special_tokens = False,
                    return_tensors="pt"
                )

                all_input_ids.append(tok["input_ids"].squeeze(0))
                all_attention_masks.append(tok["attention_mask"].squeeze(0))
        
            all_input_ids.append(torch.tensor([self.tokeniser.sep_token_id]))
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

    def __len__(self) -> int:
        """
        Gets the length of the samples to be processed.
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict: 
        """
        Retrieves a sample at a specific index.

        Arguments:
            idx (int) = The integer for the specific index position.

        Returns:
            self.samples[idx] = The samples at a given index position.
        """
        return self.samples[idx]
   
