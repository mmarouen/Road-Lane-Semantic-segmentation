import os
import argparse
from keras.optimizers import Adam
import matplotlib.pyplot as plt
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

    # build model
    if config.model_name =="fcn8":
        model_version = 8
    fcn_model = FcnModel(config.n_class, config.input_height, config.input_width, config.weights_path)
    fcn_model.model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4),metrics=['acc'])
    history = fcn_model.model.fit(x=X_train, y=y_train, epochs=10, batch_size=10, validation_data=(X_val, y_val))

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    fcn_model.model.save("fcn32.h5")


def main():
    args = parse_arguments()
    train_config = ConfigReader(args.config)
    train_model(train_config)

if __name__ == '__main__':
    main()