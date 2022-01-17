import uuid
import os
import numpy as np
import tensorflow as tf
import argparse
import valohai

VH_OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', '.outputs/')
VH_INPUTS_DIR = os.getenv('VH_INPUTS_DIR', '.inputs/')

def log_metadata(epoch, logs):
    """Helper function to log training metrics"""
    with valohai.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    return parser.parse_args()

def main():
    # valohai.prepare enables us to update the valohai.yaml configuration file with
    # the Valohai command-line client by running `valohai yaml step train_model.py`

    # valohai.prepare(
    #     step='train-model',
    #     image='tensorflow/tensorflow:2.6.0',
    #     default_inputs={
    #         'dataset': 'https://valohaidemo.blob.core.windows.net/mnist/preprocessed_mnist.npz',
    #     },
    #     default_parameters={
    #         'learning_rate': 0.001,
    #         'epochs': 5,
    #     },
    # )

    # Read input files from Valohai inputs directory
    # This enables Valohai to version your training data
    # and cache the data for quick experimentation

    #input_path = valohai.inputs('dataset').path()
    args = parse_args()

    mnist_file_path = os.path.join(VH_INPUTS_DIR, 'dataset/preprocessed_mnist.npz')

    with np.load(mnist_file_path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Print metrics out as JSON
    # This enables Valohai to version your metadata
    # and for you to use it to compare experiments

    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
    model.fit(x_train, y_train, epochs=args.epochs, callbacks=[callback])

    # Evaluate the model and print out the test metrics as JSON

    test_loss, test_accuracy = model.evaluate(x_test,  y_test, verbose=2)
    with valohai.logger() as logger:
        logger.log('test_accuracy', test_accuracy)
        logger.log('test_loss', test_loss)

    # Write output files to Valohai outputs directory
    # This enables Valohai to version your data
    # and upload output it to the default data store

    suffix = uuid.uuid4()

    #output_path = valohai.outputs().path(f'model-{suffix}.h5')
    save_path = os.path.join(VH_OUTPUTS_DIR, f'model-{suffix}.h5')

    model.save(save_path)


if __name__ == '__main__':
    main()
