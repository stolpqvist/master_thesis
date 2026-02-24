import pandas as pd
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

        for row in self.row_yielder():

            row = [['TilldeladBeredningsgruppKortNamn', "AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "BeskrivningEng", "Nyckelord"]]
            
            label_extractor(row['TilldeladBeredningsgruppKortNamn'])

            #tokenized text (tuple/list)
            list_tokenized_text = feature_extractor(row[["AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "BeskrivningEng", "Nyckelord"]])

        def label_extractor(label: str) -> None:

            label_id = self.label2id[label]
            self.id2label[label_id] = label
            

        def feature_extractor(row):
            """
            Extract each as 5 different fields
            """
            pass