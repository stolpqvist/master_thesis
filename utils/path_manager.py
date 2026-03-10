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
    
    
    

#pm = PathManager()
#path = pm.get_group_csv("NT")  
# master_thesis/datasets/NT/NT_dataset.csv