from dataclasses import dataclass


@dataclass
class Config:
    model:      str,
    k:          int,
    batch_size: int,
    n_epochs:   int,
    lr:         float,
    dropout:    float,
    columns:    InitVar[list | str]
    
