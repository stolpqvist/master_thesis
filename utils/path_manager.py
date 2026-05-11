"""
This modules handles the folder/directory creation and dataset/model retreival.
"""
from pathlib import Path

class PathManager:
    """
    This class deals with checking whether the required directories are already present, if not it
    proceeds to create them. It also deals with retreiving the models and dataset from the correct directories
    depending on user input.

    Attributes:
        root (str) = The root directory path to the master_thesis directory.
        datasets_dir (str) = The root directory path to the datasets.
        tokeniser_dir (str) = The root directory path to the tokeniser directory.
        models (str) = The root directory path to the model directory.
        results (str) = The root directory path to the results directory.
        boot (str) = The root directory path to the boot directory.

    Methods:
        setup()
            Creates a directory for all datasets   
        get_group_dir(group_name:str)
            Creates a directory for all research subjects.
        get_dataset_csv(group_name:str)
            Retrieves the path to a particular dataset depending on user input.
        get_trainval_csv(group_name: str)
            Retrieves the path to a particular training/validation file.
        get_test_csv(group_name: str)
            Retrieves the path to a particular test file.
        setup_tok()
            Creates a tokeniser directory.
        get_tok(model_name: str)
            Retrieves a particular tokeniser model.
        setup_result(model_name:str)
            Creates a result directory.
        setup_model(list_model: list[str])
            Creates model directories.
        get_model(model_name: str, bg: str)
            Retrieves a particular model trained on a particular dataset.
        setup_boot()
            Creates a boot directory in results for bootstrapping.
    """

    def __init__(self, root: str | Path = "." ):
        #pm = PathManager("/home/user/project")  string
        #pm = PathManager(Path("/home/user/project")) path object

        self.root = Path(root).resolve() #"/master_thesis"
         #convert to a Path object and figuring where it is kinda

        self.datasets_dir = self.root / "datasets"
        self.tokenizer_dir = self.root / "preprocessing" / "tokenizers"
        self.models = self.root / "models"
        self.results = self.root / "results"
        self.boot = self.results /"boot"
 
    
    def setup(self) -> None:
        """
        This method creates a directory for all datasets.
        """
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
       
    def get_group_dir(self, group_name: str) -> str:
        """
        This method creates a directory for all research subjects,
        the subdirectories are created within the datasets directory.
        It also returns the path to that particular group.

        Arguments:
            group_name (str) = The particular group name to be used for directory creation.

        Returns:
            path (str) = The path to the particular group directory.
        """
        path= self.datasets_dir / group_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_dataset_csv(self, group_name: str) -> str:
        """
        This method retrieves a particular dataset for a particular research subject.

        Arguments:
            group_name (str) = The particular group name from which to retrieve the dataset.

        Returns:
            str = The particular path from root to the dataset depending on user input.
        """
        return self.datasets_dir / group_name / f"{group_name}_dataset.csv"
    
    def get_trainval_csv(self, group_name: str) -> str:
        """
        This method retrieves a particular training and validation dataset.
        
        Arguments:
            group_name (str) = The particular group name from which to retrieve the training and validation set.

        Return:
            str = The path to a particular dataset file for a given group.
        """
        return self.datasets_dir / group_name / f"{group_name}_trainval.csv"
    
    def get_test_csv(self, group_name:str) -> str:
        """
        This method retrieves the test set for a particular group.

        Arguments:
            group_name (str) = The name of a particular group to retrieve the test dataset from.

        Returns:
            str = The path to a particular test dataset from a particular group.
        """
        return self.datasets_dir / group_name / f"{group_name}_test.csv"
        
    def setup_tok(self) -> None:
        """
        This method creates a directory to store the tokeniser in.
        """
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    def get_tok(self, model_name: str) -> str:
        """
        This method checks whether the tokeniser path to a particular tokeniser exists or not.

        Arguments:
            model_name (str) = The name of a particular tokeniser model.

        Returns:
            bool = True | False depending on whether the directory exists.
            
        """
        path = self.tokenizer_dir / f"{model_name}.model"

        return path.exists()
    
    def setup_result(self, model_name: str) -> None:
        """
        This method creates the results directories for the model, one for text, one for images.

        Arguments:
            model_name (str) = The name of the particular model used.
        """

        dirs = ['text', 'images']

        for d in dirs:
            results = self.results / model_name / d
            results.mkdir(parents=True, exist_ok=True)
    
    def setup_model(self, list_model: list[str]) -> None:
        """
        This method creates trained model directories for the models sent.

        Arguments:
            list_model (list(str)) = The list of models chosen.
        """
        if isinstance(list_model, str):
            list_model = list_model.split()


        for model in list_model:
            path = self.models / model
            path.mkdir(parents=True, exist_ok=True)
    
    def get_model(self, model_name: str, bg: str) -> str:
        """
        This method retrieves a particular model trained on a particular dataset.

        Arguments:
            model_name (str) = The name of the model to retrieve.
            bg (str) = The name of the group that the model has been trained on.

        Returns:
            str = The root path to the particular model.
        """
        return self.models / f"{model_name}" / f"{model_name}_{bg}.pt"
    
    def setup_boot(self) -> None:
        """
        This methods creates the boot directory in results in which to store all the bootstrapping and model
        comparison tests.
        """
        print("We are in setup_boot")
        self.boot.mkdir(parents=True, exist_ok=True)

