import main
import os
import numpy as np
from run import Runner 

class GridSearch():
    def __init__(self, es, lrs, wds, hs, bs, base=None, act='relu'):
        self.es = es
        self.lrs = lrs
        self.wds = wds
        self.hs = hs
        self.bs = bs
        self.activation = act
        self.base = base

        self.acc = []


    def search(self):
        for e in self.es:
            for lr in self.lrs:
                for wd in self.wds:
                    for h in self.hs:
                        for b in self.bs:
                            paras = [e, lr, wd, b, 'cosine']
                            runner = Runner(
                                h=h, 
                                activation=self.activation, 
                                paras=paras, 
                                extract_path=None, 
                                save=True,
                                base = self.base,
                                vis_choice = [True, False, False] #loss, weight, confusion,
                                )
                            acc, model = runner.run()
                            self.acc.append(acc)


def search():
    base_dir = './gridsearch'
    es = [100, 200, 300]
    lrs = [0.1, 0.13, 0.16]
    wds = [1e-5, 5e-5, 1e-4]
    hs = [100, 128, 150]
    bs = [50, 100, 200, 500]

    grid_search = GridSearch(es, lrs, wds, hs, bs, base_dir)
    grid_search.search()

    print(grid_search.acc)
    print("The best model is saved to:", )


search()