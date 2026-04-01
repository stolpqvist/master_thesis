from pathlib import Path

class PathManager:
    """
    Create the folders
    """

    def __init__(self, root: str | Path = "." ):
        #pm = PathManager("/home/user/project")  string
        #pm = PathManager(Path("/home/user/project")) path object

        self.root = Path(root).resolve() #"/master_thesis"
         #convert to a Path object and figuring where it is kinda

        self.datasets_dir = self.root / "datasets"
        self.tokenizer_dir = self.root / "preprocessing" / "tokenizers"
        self.models = self.root / "model"
        self.results = self.root / "results"
        self.boot = self.results /"boot"
 
    
    def setup(self):
        """
        create a dir for all datssets
        """
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
       
    def get_group_dir(self, group_name:str):
        """
        create dirs for specific groups
        """
        path= self.datasets_dir / group_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_dataset_csv(self, group_name: str):
        """
        get a dataset csv
        """
        return self.datasets_dir / group_name / f"{group_name}_dataset.csv"
    
    def get_trainval_csv(self, group_name: str):
        """
        train/val csv saved
        """
        return self.datasets_dir / group_name / f"{group_name}_trainval.csv"
    
    def get_test_csv(self, group_name:str):
        """
        test cav saved
        """
        return self.datasets_dir / group_name / f"{group_name}_test.csv"
        
    def setup_tok(self):
        """
        create a dir for tokenizers
        """
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    def get_tok(self, model_name):
        """
        check if tok model is avaialable
        """
        path = self.tokenizer_dir / f"{model_name}.model"

        return path.exists()
    
    def setup_result(self, model_name:str):
        """
        set up dirs for saving a model and images
        """

        dirs = ['text', 'images']

        for d in dirs:
            results = self.results / model_name / d
            results.mkdir(parents=True, exist_ok=True)
    
    def setup_model(self, model_name):
        """
        creates a folder for saving models
        """
        self.models.mkdir(parents=True, exist_ok=True)
    
    def get_model(self, model_name):
        """
        retrieve the model
        """
        return self.models / f"{model_name}.pt"
    
    def setup_boot(self):
        """
        set up dir for saving bootstrap 
        """
        print("We are in setup_boot")
        self.boot.mkdir(parents=True, exist_ok=True)


#pm = PathManager()
#path = pm.get_group_csv("NT")  
# master_thesis/datasets/NT/NT_dataset.csv
