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
import torch
from transformers import AutoTokenizer
from collections import defaultdict





class DataProcessor:
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
    def __init__(self, filename):
        self.file = filename
        self.label2id = defaultdict(lambda: len(self.label2id))
        self.id2label = {}
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    def row_yielder(self):
        """
        Reads the class instance of self.file, opens the csv file and yields the file
        row by row.

        Return/Yields:
            row, list[str] = the data of the dataframe from a particular row
        """
        df = pd.read_csv(self.file)
        for _, row in df.iterrows():
            yield row
    
    def preprocessing(self):
       """
        Handles the logic of preprocessing the data from the provided dataframe.
        It calls row yielder and then processes it with 2 inner functions.
        
        Inner functions:
            label_extractor(str): extract and handles the label logic of assigning each label a value
                                  and each value a label for label identification.

            auto_tok(str): tokenizes a given column and returns it as a tensor.

        Returns:
            list_tok(list(torch.Tensor)): all the extracted and tokenized columns as a list.
       """ 
        for row in self.row_yielder():

            def label_extractor(label: str) -> None:
                """
                Extracts the labels and enters it into a defauldict to handle label -> id and
                id -> label assignment.
                """

                label_id = self.label2id[label] 
                self.id2label[label_id] = label
            
            def auto_tok(column: str) -> torch.Tensor: 
                """
                Takes a given column as a string and returns a tokenized torch.Tensor
                """
                return self.tokenizer(column, return_tensors="pt")

           # def stacker(list_pt:list[torch.Tensor]) -> torch.Tensor:
           #     return torch.stack(list_pt, dim=0)


            row = row[['TilldeladBeredningsgruppKortNamn', "AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "BeskrivningEng", "Nyckelord"]]
            
            label_extractor(row['TilldeladBeredningsgruppKortNamn'])

            #tokenized text (tuple/list)
            #list_tokenized_text = feature_extractor(row[["AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "BeskrivningEng", "Nyckelord"]])
            to_tokenize = row.drop('TilldeladBeredningsgruppKortNamn')
            #for each column in a row -> tokenize -> list of tesors of tokenized texts
            list_tok = [auto_tok(column) for column in to_tokenize]

            
            #print(f"Tokenised: {len(list_tok)}")
            return list_tok


if __name__ == "__main__":
    import sys
    file = sys.argv[1]
    dp = DataProcessor(file)
    dp.preprocessing()
    
