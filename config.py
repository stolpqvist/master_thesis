from dataclasses import dataclass
from enum import Enum

class Model(Enum):
    roberta = "roberta"
    rnn = "rnn"
    cnn = "cnn"



@dataclass
class Config:
    model:      Model
    k:          int
    batch_size: int
    n_epochs:   int
    lr:         float
    dropout:    float
    columns:    list

    def __post_init__(self):
        if not isinstance(self.model, Model):
            raise ValueError(f"Invalid model. Choose from: {[m.value for m in Model]}")

    
#Usage:
#config = Config(
#   model=Model.roberta,
#   k=5,
#   batch_size=3,
#   n_epochs = 10,
#   lr=0.001,
#   dropout=0.3,
#   columns=["col1", "col2"])