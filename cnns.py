from tensorflow import keras
from keras import datasets, layers, models

from keras.models import load_model, save_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, BatchNormalization, Dropout, Activation, add, AveragePooling2D
import tensorflow as tf

def build_model(hp):
    model = keras.Sequential()

    model.add(layers.Conv2D(
        filters=hp.Choice('filters_1', values=[32, 64]),
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(256, 224, 12)
    ))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(
        filters=hp.Choice('filters_2', values=[64, 128]),
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(
        filters=hp.Choice('filters_3', values=[64, 128]),
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(layers.Flatten())

    model.add(layers.Dense(
        units=hp.Int('units', min_value=32, max_value=128, step=32),
        activation='relu'
    ))

    model.add(layers.Dense(1))

    # compile the model
    model.compile(optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae'])
    return model

def create_lstm_cnn_model(hp): # Tuning the CNN, LSTM layers, kernel size, learning rate, optimizer and the dropouts
    input_shape = (4, 256, 224, 3)
    model = keras.Sequential()
    #ks = hp.Int("kernel_size", min_value=2, max_value=5)

    # Convolutional layers
    for i in range(hp.Int('conv_layers', min_value=1, max_value=3)):
        model.add(TimeDistributed(Conv2D(hp.Int('filters_' + str(i), min_value=16, max_value=64, step=16),
                                         (3, 3), activation='relu'), input_shape=input_shape))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Dropout(0.5)))

    # Flatten layer
    model.add(TimeDistributed(Flatten()))

    # LSTM layers
    lstm_units = hp.Int('lstm_units', min_value=16, max_value=64, step=16)
    model.add(LSTM(lstm_units, dropout=hp.Float("lstm_dropout", min_value=0.3, max_value=0.7),
                   recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.2, max_value=0.6),
                   return_sequences=False))

    model.add(Dense(1, activation='relu'))

    opt = hp.Choice("activation", ["adam", "sgd", 'rmsprop', "adagrad"])
    learning_rate = hp.Float("lr", min_value=1e-3, max_value=1e-2, sampling="log")

    if opt == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif opt == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif opt == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model

def alexnet(hp): # Tuning the kernel size, optimizers, learning rate
  model = models.Sequential()
  #filter_sizes = [hp.Int(f"filters_{i}", min_value=2, max_value=5) for i in range(2, 5)]
  ks = hp.Int("kernel_size", min_value=2, max_value=3)

# Use the list in the model definition
  #ks = hp.choice("filters", filter_sizes)
  model.add(layers.Conv2D(32, ks , padding='same' ,input_shape=(256, 224, 12)))
  model.add(layers.Activation('relu'))
  model.add(layers.MaxPooling2D(3, strides=2))
  # model.add(BatchNormalization())

  model.add(layers.Conv2D(64, ks, padding='same'))
  model.add(layers.Activation('relu'))
  model.add(layers.MaxPooling2D(3))
  # model.add(BatchNormalization())

  model.add(layers.Conv2D(128, ks, padding='same'))
  model.add(layers.Activation('relu'))

  model.add(layers.Conv2D(128, ks, padding='same'))
  model.add(layers.Activation('relu'))

  model.add(layers.Conv2D(64, ks,  padding='same'))
  model.add(layers.Activation('relu'))

  model.add(layers.Flatten())
  model.add(layers.Dense(400, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dropout(0.5))
  opt = hp.Choice("activation", ["adam", "sgd",'rmsprop','adagrad'])
  model.add(layers.Dense(1, activation='relu'))
  learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
  if opt == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  elif opt == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif opt == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
  elif opt == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
  model.compile(optimizer = optimizer,loss ='mean_squared_error',metrics = ['mae'])
  return model


def Conv2D_BN(x, nb_filter, kernel_size, strides=1, padding='same', name=None):
    x = Conv2D(nb_filter, kernel_size, padding='same', data_format='channels_last', strides=strides,
               activation='relu')(x)
    x = BatchNormalization()(x)
    return x

def identity_Block(inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False, padding='same'):
    x = Conv2D_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2D_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2D_BN(inpt, nb_filter=nb_filter, strides=strides,
                             kernel_size=kernel_size, padding='same')
        shortcut = Dropout(0.2)(shortcut)  # Move dropout to the shortcut branch
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def resnet(hp):
    inpt = Input(shape=(256,224,12))
    #ks = hp.Int("kernel_size", min_value=2, max_value=5)

    for i in range(1, hp.Int("num_layers", 1, 2) + 1):
        if i == 1:
            x = Conv2D_BN(inpt, nb_filter=16 * i**2, kernel_size=(3, 3), strides=1, padding='same')
            x = MaxPooling2D(pool_size=(2, 2), strides=2, data_format='channels_last')(x)
        else:
            x = identity_Block(x, nb_filter=32 * i**2, kernel_size=(3, 3), strides=1, with_conv_shortcut=True, padding='same')
            x = identity_Block(x, nb_filter=32 * i**2, kernel_size=(3, 3), strides=1)
            x = identity_Block(x, nb_filter=32 * i**2, kernel_size=(3, 3))

    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='relu')(x)
    model = Model(inputs=inpt, outputs=x)
    learning_rate = hp.Float("lr", min_value=1e-3, max_value=1e-2, sampling="log")
    opt = hp.Choice("activation", ["adam", "sgd"])
    if opt == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # learning_rate=learning_rate
    elif opt == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model