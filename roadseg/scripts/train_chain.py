import os
import argparse
from roadseg.utils.config_loader import ConfigReader
from roadseg.utils.data_utils import DataUtils
from roadseg.models.fcn_models import FcnModel

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train road image segmentation model")
    parser.add_argument('--config','-c', help='Path to the configuration file', required=True)

    return parser.parse_args()

def train_model(config: ConfigReader):

    # load data
    data_handler = DataUtils(config.data_path, config.weights_path)
    X_train,y_train = data_handler.get("data/training", config.n_class, config.input_height, config.input_width)
    if config.augment_data:
        X_train,y_train = data_handler.augment_training_data(X_train,y_train, config.n_class, config.input_height, config.input_width)
    print("training data shape: {0}".format(X_train.shape))

    X_val, y_val = data_handler.get("data/validation", config.n_class, config.input_height, config.input_width)
    print("validation data shape: {0}".format(X_val.shape))

    # build and save model
    if config.model_name =="fcn8":
        model_version = 8
    fcn_model = FcnModel(config.n_class, config.input_height, config.input_width, config.weights_path)
    history = fcn_model.fit(X_train, y_train, X_val, y_val, config.optimizer, config.lr)
    model_path = os.path.join(os.getcwd(),"models")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    fcn_model.model.save(os.path.join(model_path,"fcn32.h5"))

    plot_path = os.path.join(os.getcwd(),"plots")
    fcn_model.plot_perf(history, plot_path)

def main():
    args = parse_arguments()
    train_config = ConfigReader(args.config)
    train_model(train_config)

if __name__ == '__main__':
    main()