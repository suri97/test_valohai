import numpy as np
#import valohai
import os

VH_OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', '.outputs/')
VH_INPUTS_DIR = os.getenv('VH_INPUTS_DIR', '.inputs/')

def main():
    # valohai.prepare enables us to update the valohai.yaml configuration file with
    # the Valohai command-line client by running `valohai yaml step preprocess_dataset.py`

    # valohai.prepare(
    #     step='preprocess-dataset',
    #     image='python:3.9',
    #     default_inputs={
    #         'dataset': 'https://valohaidemo.blob.core.windows.net/mnist/mnist.npz',
    #     },
    # )

    # Read input files from Valohai inputs directory
    # This enables Valohai to version your training data
    # and cache the data for quick experimentation

    print('Loading data')
    mnist_file_path = os.path.join(VH_INPUTS_DIR, 'dataset/mnist.npz')

    with np.load(mnist_file_path, allow_pickle=True) as file:
        x_train, y_train = file['x_train'], file['y_train']
        x_test, y_test = file['x_test'], file['y_test']

    print('Preprocessing data')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Write output files to Valohai outputs directory
    # This enables Valohai to version your data
    # and upload output it to the default data store

    print('Saving preprocessed data')
    save_path = os.path.join(VH_OUTPUTS_DIR, 'preprocessed_mnist.npz')
    np.savez_compressed(save_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    main()
