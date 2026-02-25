import pandas as pd
import torch
from transformers import XLMRobertaForSequenceClassification, AutoTokenizer
from collections import defaultdict


class DataProcessor:
    def __init__(self, filename):
        self.file = filename
        self.label2id = defaultdict(lambda: len(self.label2id))
        self.id2label = {}
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    def row_yielder(self):

        df = pd.read_csv(self.file)
        for _, row in df.iterrows():
            yield row
    
    def preprocessing(self, row):
        
        #TODO Do we want to yield it row by row, and that each item in the list is
        #a row? Or do we want to simply process it row by row and train it as such
        #TODO Figure out training sequence
        for row in self.row_yielder():

            row = [['TilldeladBeredningsgruppKortNamn', "AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "BeskrivningEng", "Nyckelord"]]
            
            label_extractor(row['TilldeladBeredningsgruppKortNamn'])

            #tokenized text (tuple/list)
            list_tokenized_text = feature_extractor(row[["AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "BeskrivningEng", "Nyckelord"]])
            

        def label_extractor(label: str) -> None:

            label_id = self.label2id[label]
            self.id2label[label_id] = label
            

        def feature_extractor(row:list) -> list:
            """
            Extract each as 5 different fields
            """
            tok_list = []
            for column in row:
                tok_list.append(column)
            return tok_list

        def auto_tok(tok_list:list) -> torch.Tensor:
            return self.tokenizer(tok_list)
