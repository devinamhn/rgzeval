from pathlib import Path


class Path_Handler:
    """Handle and generate paths in project directory"""

    def __init__(
        self, **kwargs
    ):  # use defaults except where specified in kwargs e.g. Path_Handler(data=some_alternative_dir)
        path_dict = {}
        path_dict["root"] = kwargs.get("root", Path(__file__).resolve().parent.parent.parent)
        path_dict["project"] = kwargs.get(
            "project", Path(__file__).resolve().parent
        )  # i.e. this repo

        path_dict["data"] = kwargs.get("data", path_dict["project"] / "data")

        for key, path_str in path_dict.copy().items():
            path_dict[key] = Path(path_str)

        self.path_dict = path_dict

    def fill_dict(self):
        """Create dictionary of required paths"""

        self.path_dict["rgz"] = self.path_dict["data"] #/ "rgz108k-batches-py" #rgz
        # self.path_dict["mb"] = self.path_dict["data"] / "mb"
        # self.path_dict["mightee"] = self.path_dict["data"] / "MIGHTEE"

    def create_paths(self):
        """Create missing directories"""
        for path in self.path_dict.values():
            create_path(path)

    def _dict(self):
        """Generate path dictionary, create any missing directories and return dictionary"""
        self.fill_dict()
        self.create_paths()
        return self.path_dict


def create_path(path):
    if not Path.exists(path):
        Path.mkdir(path)
