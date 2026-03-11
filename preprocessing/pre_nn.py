import sentencepiece as spm
from utils.path_manager import PathManager as pm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

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
    def __init__(self, df, model='tokenizer', vocab='vocabulary'):

        self.pm = pm("./")

        if self.checker(model) is False:
            self.create_vocab(df, model)
        self.model = self.get_vocab(model)
        #print(self.model)

    def checker(self, model):
        #check if model already exists

        if not pm.get_tok(model):
            return False
    
    def create_vocab(self, df, model):
        """
        1. load file -> prepare the input for the Trainer 
        2. train the tokenizer (prepare vocab)
        """
        

        text_columns = ["AnsökanTitel", "AnsökanTitelEng", "Beskrivning", "Nyckelord"]

        all_text = pd.concat([df[col].dropna() for col in text_columns], ignore_index=True)

        pm.setup_tok()

        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(all_text.astype(str)), #input tp construct vocab
            model_prefix = model,
            vocab_size = 16000,
            character_coverage=1.0, #all unique characters
            model_type = 'bpe', 
            minloglevel=2
        )
        
    
    def get_vocab(self, model):
        #load the tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(f'{pm.tokenizer_dir}/{model}.model')
        return sp

    
    def tokenizer(self, text):

        #Example
        #text = "Nuclear reactions are common in organic chemistry"
        token_ids = self.model.encode(text, add_bos=True, add_eos=True, out_type=int)
        #print(token_ids)

        #Subword pieces if you're curious
        #pieces = self.model.encode(text, out_type=str)
        #print(pieces)

        #NOTE: to decode : sp.decode(ids)
        return token_ids #, pieces


#if __name__ == "__main__":
#    df = pd.read_csv("./datasets/NT/NT_dataset.csv")
#    sp = SPTokenizer(df)
#    sp.tokenizer()



