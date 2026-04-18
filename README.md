# Fashion-MNIST Image Classification with Neural Network

A three-layer neural network implementation for Fashion-MNIST classification, featuring:

- Training with Mini-Batch GD
- Hyperparameter search
- Learning rate scheduling
- Training visualization
- Autosave for paras & model & results

## Project Structure
- dataloader.py: load dataset from ubyte file
- model.py: model definition of the 3-layer NN, including weights visualization
- train.py: trainer for model, including loss/accuracy curves visualization
- test.py: validator for model,including confusion-matrix visualization
- para_search.py: grid search agent for paras search
- run.py: the main integrator of all agents
- main.py: entrance of the code allowing for paras setting and quick start&save
- complete_code.py: the entire code allows for quick settings and one-button-start 
- data: folder that contains raw fashion-MNIST dataset
- README.md

## Installation

1.Clone repository:
```bash

```

2.Install dependencies:
```bash
!pip install numpy matplotlib os math
```

## Dataset Preparation
1.Download fashion-MNIST dataset from website

2.Extract files to ./data and maintain this structure:
- data/
    ├── train-images-idx3-ubyte
  
    ├── train-labels-idx1-ubyte
  
    ├── t10k-images-idx3-ubyte
  
    ├── t10k-labels-idx1-ubyte


## Usage
1.For a one-tine quick start, you can change paras in complete_code.py(shown as follows):
```python
hidden_layer = 150
paras = [100, 0.1, 1e-4, 50, 'cosine']
activation = 'relu'
save = True
origin_path = None
```

In main.py you can run code in the same way.


- For training in a larger scale and para-search, use para_search.py for grid search


- For importing a already trained model for validation, set origin_path as your file path:
