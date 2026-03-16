import sentencepiece as spm
from utils.path_manager import PathManager as pm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

#NOTE check the hierarchy of the files ad folders


class SPTokenizer:
    """
    1. check if the .vocabulary already exists 
    2. if not -> prepare text -> send it to tokenizer.train
    2.5. if yes -> load .vocabulary

    3. Load the tokenizer
    4. Tokenize

    Functions:

    def checker() #if the vocab exists
    def create_vocab()
    def get_vocab() #load 
    def tokeniser()

    if checker() is False:
        create_vocab()
    get_vocab()
    tokenise() -> tokenised text

    
    """
    def __init__(self, df=None,  model='tokenizer', vocab='vocabulary', label="TilldeladBeredningsgruppKortNamn"):
        
        self.df = df
        self.label = label
        self.pm = pm("./")
        self.label2id = {} 
        self.id2label = {}

        if self.df is not None:
            if self.checker(model) is False:
                self.create_vocab(model)

        #load the model        
        self.model = self.get_vocab(model)
        

        


    def label_extractor(self, df) -> None:
        """
        Extracts the labels and enters it into a defauldict to handle label -> id and
        id -> label assignment.
        """

        #add to labels
        for i, label in enumerate(np.unique(df[self.label].values)):
            self.label2id[label] = i
            self.id2label[i] = label

    def checker(self, model):
        #check if model already exists

        if not self.pm.get_tok(model):
            return False
        return True
    
    def create_vocab(self, model):
        """
        1. load file -> prepare the input for the Trainer 
        2. train the tokenizer (prepare vocab)
        """

        text_columns = ["AnsökanTitel", "AnsökanTitelEng", "Sammanfattning", "Populärbeskrivning", "Nyckelord"] 
        all_text = pd.concat([self.df[col].dropna() for col in text_columns], ignore_index=True)

        self.pm.setup_tok()

        #print(f"[DEBUG] tokenizer_dir exists: {self.pm.tokenizer_dir.exists()}")  
    
        #model_prefix = str(self.pm.tokenizer_dir / model)
        #print(f"[DEBUG] saving model to: {model_prefix}")

        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(all_text.astype(str)), #input tp construct vocab
            model_prefix = str(self.pm.tokenizer_dir / model),
            vocab_size = 16000,
            character_coverage=1.0, #all unique characters
            model_type = 'bpe', 
            minloglevel=2
        )

        #expected = self.pm.tokenizer_dir / f"{model}.model"
        #print(f"[DEBUG] file created: {expected.exists()}")
        
    
    def get_vocab(self, model):
        #load the tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(str(self.pm.tokenizer_dir/f"{model}.model"))

        #print("Tokenizer model loaded") prints 2 times
        
        return sp

    
    def tokenizer(self, df):

        #Example
        #text = "Nuclear reactions are common in organic chemistry"
        #token_ids = self.model.encode(text, add_bos=True, add_eos=True, out_type=int)
        #print(token_ids)

        #Subword pieces if you're curious
        #pieces = self.model.encode(text, out_type=str)
        #print(pieces)

        text_columns = ["AnsökanTitel", "AnsökanTitelEng", "Sammanfattning", "Populärbeskrivning", "Nyckelord"] 
        tokens = []
        labels = []

        for _, row in df.iterrows():

            all_text = " ".join([str(row[col]) for col in text_columns if pd.notna(row[col])])
            label = row[self.label]


            token_ids = self.model.encode(all_text, add_bos=True, add_eos=True, out_type=int)
            label_idx = self.label2id[label]

            tokens.append(torch.tensor(token_ids))
            labels.append(torch.tensor(label_idx))
        
        t_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
        #t_tokens = torch.stack(tokens).unsqueeze(1)
        t_labels = torch.stack(labels)#.unsqueeze(1)

        return t_tokens, t_labels



                

        #NOTE: to decode : sp.decode(ids)
        #return token_ids #, pieces


#if __name__ == "__main__":
#    df = pd.read_csv("./datasets/NT/NT_dataset.csv")
#    sp = SPTokenizer(df)
#    sp.tokenizer()



