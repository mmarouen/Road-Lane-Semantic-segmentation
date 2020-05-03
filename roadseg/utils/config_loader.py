import json

class ConfigReader():
    def __init__(self, filename):
        self.config = self.read_file(filename)
    
    def read_file(self, filename):
        with open(filename, "r") as f:
            return json.load(f)
    
    @property
    def data_path(self):
        return self.config["data_path"]

    @property
    def weights_path(self):
        return self.config["weights_path"]
    
    @property
    def augment_data(self):
        return self.config["augment_data"]
    
    @property
    def model_name(self):
        return self.config["model_name"]

    @property
    def n_class(self):
        return self.config["n_classes"]

    @property
    def input_width(self):
        return self.config["input_width"]

    @property
    def input_height(self):
        return self.config["input_height"]
