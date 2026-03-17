from dataclasses import dataclass
from enum import Enum

class Model(Enum):
    roberta = "roberta"
    rnn = "rnn"
    cnn = "cnn"



@dataclass
class Config:
    model:      str,
    k:          int,
    batch_size: int,
    n_epochs:   int,
    lr:         float,
    dropout:    float,
    columns:    InitVar[list | str]

    def __post_init__(self):
        if not isinstance(self.model, Model):
            raise ValueError(f"Invalid model. Choose from: {[m.value for m in Model]}")

    
