"""
This module deals with input data and making sure it is of the correct type.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Union

class Model(Enum):
    roberta = "roberta"
    rnn = "rnn"
    cnn = "cnn"



@dataclass
class Config:
    """
    This class is a dataclass and checks the type of every value passed through this class, to check the type and whether extra
    values have been passed.

    Attributes:
        model (list(str)) = The name of the models to be used.
        k (int) = The amount of K-folds to split the data in.
        batch_size (int) = The amount of batches for the data, i.e how much data to be loaded at once.
        n_epochs (int) = The amount of epochs to be trained over.
        lr (float) = The learning rate to be applied to the model.
        drouput (float) = The dropout rate to be applied to the model.
        bg (str) = The name of the subject matter group to be trained/tested on.
        columns (list(str)) = The headers of columns to be used for information in training/testing.
        label (str) = The label header to be used to compare the model's predictions against.
        create_data (bool) = Boolean that controls whether to create a new dataset.
        param_hunt (bool) =  Boolean that controls whether to do hyperparameter optimisations.
        train (bool) = Boolean that controls whether to do training or not.
        test (bool) = Boolean that controls whether or not to do testing.
        boot (bool) = Boolean that controls whether or not to do bootstrapping.
        vis (bool) = Boolean that controls whether or not to visualise the results or not.
        emissions (bool) = Boolean that controls whether or not to track emissions.
    """
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
        """
        This method controls whether the selected model exists in available models.

        Arguments:
            cls = The config dataclass.
            args = The arguments passed from the config class.

        Returns:
            cls = The dataclass values.
        """
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
    

