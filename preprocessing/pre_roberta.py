import pandas as pd
import torch
from transformers import XLMRobertaForSequenceClassification, AutoTokenizer
from collections import defaultdict
import sys

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
    
    def preprocessing(self):
        
        #TODO Do we want to yield it row by row, and that each item in the list is
        #a row? Or do we want to simply process it row by row and train it as such
        #TODO Figure out training sequence
        for row in self.row_yielder():

            def label_extractor(label: str) -> None:

                label_id = self.label2id[label]
                self.id2label[label_id] = label
            
            def auto_tok(column: str) -> torch.Tensor: 
                #print(f"The text column is: {column}")
                return self.tokenizer(column, return_tensors="pt")


            row = row[['TilldeladBeredningsgruppKortNamn', "AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "BeskrivningEng", "Nyckelord"]]
            
            label_extractor(row['TilldeladBeredningsgruppKortNamn'])

            #tokenized text (tuple/list)
            #list_tokenized_text = feature_extractor(row[["AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "BeskrivningEng", "Nyckelord"]])
            to_tokenize = row.drop('TilldeladBeredningsgruppKortNamn')
            #for each column in a row -> tokenize -> list of tesors of tokenized texts
            list_tok = [auto_tok(column) for column in to_tokenize]

            #print(f"Tokenised: {len(list_tok)}")


if __name__ == "__main__":
    file = sys.argv[1]
    dp = DataProcessor(file)
    dp.preprocessing()
    
