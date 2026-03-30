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
    bg:         str
    columns:    list
    label:      str
    create_data:bool = False
    param_hunt: bool = False
    train:      bool = False
    test:       bool = False
    boot:       bool = False

    @classmethod
    def from_args(cls, args) -> "Config":
        try:
            for model in args.model:
                model = Model(model)
        except ValueError:
            raise ValueError(f"Invalid model '{args.model}'. Choose from: {[m.value for m in Model]}")
        
        return cls(
            model=      args.model,
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
            boot=       args.boot
        )
    
    def __post_init__(self):
        if not isinstance(self.model, Model):
            raise TypeError(f"model must be a Model enum, got {type(self.model).__name__}")

    

