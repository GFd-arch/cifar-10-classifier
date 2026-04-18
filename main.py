import numpy as np
import os
import matplotlib.pyplot as plt
from data_loader import idxLoader
from model import ThreeLayerNN
from run import Runner
from train import Trainer
from test import Evaluator


def main():
    hidden_layer = 150
    paras = [100, 0.1, 1e-4, 50, 'cosine']
    activation = 'relu'
    save = True
    origin_path = None
    # "./results/model_x/model_paras.npz"

    save = False if origin_path else save
    save_path = './results' if save else None

    runner = Runner(
        h=hidden_layer, 
        activation=activation, 
        paras=paras, 
        extract_path=origin_path, 
        save=save,
        base = save_path
        )
    acc, model = runner.run()
    print("Final accuracy:", acc)


if __name__ == "__main__":
    main()