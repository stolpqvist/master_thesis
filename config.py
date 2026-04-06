from dataclasses import dataclass
from enum import Enum
from typing import Union

class Model(Enum):
    roberta = "roberta"
    rnn = "rnn"
    cnn = "cnn"



@dataclass
class Config:
    model:      Union[str, list]
    k:          int
    batch_size: int
    n_epochs:   int
    lr:         float
    dropout:    float
    bg:         str
    columns:    list
    label:      str
    create_data:bool = False
    param_hunt: bool = False
    train:      bool = False
    test:       bool = False
    boot:       bool = False
    vis:        bool = False
    emissions:  bool = False

    @classmethod
    def from_args(cls, args) -> "Config":
        try:
            converted = [Model(m) for m in args.model]
        except ValueError:
            raise ValueError(f"Invalid model '{args.model}'. Choose from: {[m.value for m in Model]}")
        
        model = converted[0] if len(converted) == 1 else converted
        
        return cls(
            model=      model,
            k=          args.k,
            batch_size= args.batch_size,
            n_epochs=   args.e,
            lr=         args.lr,
            dropout=    args.dr,
            bg=         args.bg,
            columns=    args.columns,
            label=      args.label[0],
            create_data=args.create_datasets,
            param_hunt= args.param_hunt,
            train=      args.train,
            test=       args.test,
            boot=       args.boot,
            vis=        args.vis,
            emissions = args.emissions
        )
    
    def __post_init__(self):
        if isinstance(self.model, list):
            invalid = [m for m in self.model if not isinstance(m, Model)]
            if invalid:
                raise TypeError(f"All models must be Model enums, got {[type(m).__name__ for m in invalid]}")
        elif not isinstance(self.model, Model):
            raise TypeError(f"model must be a Model enum, got {type(self.model).__name__}")
    

