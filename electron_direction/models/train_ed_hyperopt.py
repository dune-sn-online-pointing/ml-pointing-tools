import sys
import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import hyperopt as hp
import inspect

sys.path.append("../python/")
import general_purpose_libs as gpl
import regression_libs as rl

# Set seed for reproducibility  
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


parser = argparse.ArgumentParser(description='Run the pipeline')
parser.add_argument('--input_json', type=str, help='Input json file')
parser.add_argument('--output_folder', type=str, help='Output folder')
args = parser.parse_args()

input_json_file = args.input_json
output_folder = args.output_folder

# Read input json
with open(input_json_file) as f:
    input_json = json.load(f)

input_data = input_json['input_data']
input_label = input_json['input_label']
model_name = input_json['model_name']
n_outputs = input_json['n_outputs']

output_folder = output_folder + model_name + '/'
output_folder = output_folder + f"aug_coeff_{input_json['dataset_parameters']['aug_coefficient']}_loss_{input_json['model_parameters']['loss_function']}_nout_{n_outputs}/"


if model_name == 'simple_cnn':
    import models.simple_cnn as selected_model
elif model_name == 'hyperopt_simple_cnn':
    import models.hyperopt_simple_cnn as selected_model
elif model_name == 'cvn_regression':
    import models.cvn_regression as selected_model
elif model_name == 'd3_views_simple_cnn':
    import models.d3_views_simple_cnn as selected_model
elif model_name == 'd3_views_hp_simple_cnn':
    import models.d3_views_hp_simple_cnn as selected_model
else:
    print('Model not found')
    sys.exit(1)

model_parameters = input_json['model_parameters']
dataset_parameters = input_json['dataset_parameters']

if __name__=='__main__':

    # Check if GPU is available
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    print('Input json file: ', input_json_file)
    print('Output folder: ', output_folder)
    
    print('Starting the pipeline')
    print('Preparing the data')
    # Prepare the data
    train, validation, test = rl.prepare_data(input_data, input_label, dataset_parameters=dataset_parameters, output_folder=output_folder)

    print('Creating the model')
    # Create the model
    model, history = selected_model.create_and_train_model( n_outputs=n_outputs, 
                                                            model_parameters=model_parameters, 
                                                            train=train, 
                                                            validation=validation,
                                                            output_folder=output_folder, 
                                                            model_name=model_name,
                                                            )
    print('Model created')

    # Save the model
    model.save(output_folder+model_name+'.h5')

    # Save the history
    gpl.save_history(history, output_folder)

    # Test the model
    print('Testing the model')
    rl.test_model(model, test, output_folder)

    # Create the report
    gpl.create_report(output_folder, model_name, input_json, inspect.getsource(selected_model.create_and_train_model))





