import os
import argparse
from roadseg.utils.config_loader import ConfigReaderFromGit
from roadseg.utils.data_utils import DataUtils
from roadseg.models.fcn_models import FcnModel

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train road image segmentation model")
    parser.add_argument('--config','-c', help='Path to the configuration file', required=True)

    return parser.parse_args()

def train_model(config: ConfigReader):

    # load data
    data_handler = DataUtils(config.data_path, config.weights_path)
    X,y = data_handler.get(config.n_class, config.input_height, config.input_width)
    if config.augment_data:
        X,y = data_handler.augment_training_data(X, y, config.n_class, config.input_height, config.input_width)
    print("training data shape: {0}".format(X.shape))

    # build model
    if config.model_name =="fcn8":
        model_version = 8
    fcn_model = FcnModel(config.n_class, config.input_height, config.input_width, config.weights_path)
    print(fcn_model.model.summary())
    
    

def main():
    args = parse_arguments()
    train_config = ConfigReaderFromGit()
    train_model(train_config)

if __name__ == '__main__':
    main()