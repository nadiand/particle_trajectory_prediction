# Deep Learning Approaches to Trajectory Reconstruction: Transformer, RNN, and Regressor

This repository contains the implementation of four deep learning models for trajectory reconstruction. It allows for generation of three different datasets, training of the models and testing. 

We implement the following:
- Regressor which directly maps hits to trajectory parameters. Can only work with data of type "dataset1", i.e. 2D data in which each event has only 3 generated secondary particles.
- Direct transformer which directly maps hits to trajectory parameters. Can only work with data of type "dataset1", i.e. 2D data in which each event has only 3 generated secondary particles.
- CL+RNN model. Uses agglomerative clustering for track finding and an RNN for track fitting. Can work with all three types of data.
- TR+RNN model. Uses a transformer for track finding and an RNN for track fitting. Can work with all three types of data.

The folder `model_structures` contains all network architectures. The folder `training` contains the files for training, evaluation and prediction of the networks. The folder `best_models` contains example trained models.

### Data 
To generate the data, simply run the `data_generation.py` file. The constants in the `global_constants.py` file decide what type of data will be generated. Relevant parameters to set: MAX_NR_TRACKS, DIM, NR_DETECTORS, EVENTS, data paths. By default, it generates a simple 2D dataset with 50,000 events, each of which having a maximum of 3 tracks, and 5 detectors. The data can also be 3D, and have any amount of tracks or events.

### Using the models
In order to run and test the CL+RNN model, simply run the `clustering_rnn.py` file. By default there is a trained RNN on the simple dataset in the `best_models` folder that will get loaded. 

In order to run and test the TR+RNN model, simply run the `transformer_rnn.py` file. By default there are trained classifying transformer and a RNN on the simple dataset in the `best_models` folder that will get loaded. 

The other two extremely simple models can be run by running the files `regressor_training.py` and `train_transformer.py` from the training folder.
