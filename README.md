# Trident 

The Trident is a collection of three super-module generators for Neural Network design and construction for biological sequences and beyond. The short paper explaining it is in progress. 

This package contains 3 modules: 

1. prongs
- This contains the super-module generators called Prongs. 

2. preass
- This contains pre-assembled neural networks ranging from simple dense NNs to convolutional autoencoders 

3. utils 
- This contains utilities used in the aforementioned modules. 
- It also contains: 
    - A special Trainer that implements useful fitting methods such as batch flipping (for biological sequences) and Epoch Sampling ([Binning Methods paper](https://doi.org/10.1101/2025.06.26.661884))
    - A Predictor for producing predictions
    - A candidate scorer and candidate repeater which are used for hyperparameter optimization with [poseigen_compass](https://github.com/Husam94/poseigen_compass)
    - Binned loss which is using bin metrics as a loss function ([Binning Methods paper](https://doi.org/10.1101/2025.06.26.661884))
    - Synthetic data generation (see [DevLoss](https://github.com/Husam94/DevLoss) case study)