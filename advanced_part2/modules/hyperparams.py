import keras_tuner

EPOCHS = 10
BATCH_SIZE = 32

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

TRAIN_LENGTH = 1034
EVAL_LENGTH = 128

INPUT_IMG_SIZE = 224

def get_hyperparameters(hyperparameters) -> keras_tuner.HyperParameters:
    hp = keras_tuner.HyperParameters()

    for hyperparameter in hyperparameters:
        hp.Choice(hyperparameter,
                  hyperparameter["values"],
                  default=hyperparameter["default"])

    return hp
