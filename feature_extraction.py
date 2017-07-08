import pickle
import tensorflow as tf
import numpy as np
# TODO: import Keras layers you need here
# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    # preprocess data
    X_normalized = np.array(X_train / 255.0 - 0.5 )

    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)
    
    # preprocess data
    X_normalized = np.array(X_train / 255.0 - 0.5 )

    # TODO: train your model here
    
    #model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    #history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
