from data_loader import idxLoader
from model import ThreeLayerNN
from train import Trainer
from test import Evaluator
import numpy as np
import os
from matplotlib import pyplot as plt


class Runner:
    def __init__(self, h, activation, paras, extract_path, save=False, base='/results',vis_choice=[True,True,True]):
        self.hidden = h
        self.activation = activation if activation is not None else 'relu'
        self.paras = paras if paras is not None else [5000, 0.14, 1e-5, 10000, 'cosine']
        self.extract_path = extract_path
        self.base_dir = base
        self.save_path = self.create_model_dir(base_dir=base) if save else None # model_x
        self.vis_choice = vis_choice # loss, weight, confusion
        self.acc = None

    def create_model_dir(self, base_dir='./results'):
        os.makedirs(base_dir, exist_ok=True)
        existing = [d for d in os.listdir(base_dir) if d.startswith('model_')]
        if not existing:
            new_id = 1
        else:
            ids = [int(d.split('_')[1]) for d in existing]
            new_id = max(ids) + 1

        new_dir = os.path.join(base_dir, f'model_{new_id}')
        os.makedirs(new_dir)
        print(f"This model will be saved to: {new_dir}")
        return new_dir

    def save_model(self, model):
        save_path = os.path.join(self.save_path, 'model_paras.npz')
        np.savez(
            save_path,
            w1 = model.paras['w1'],
            b1 = model.paras['b1'],
            w2 = model.paras['w2'],
            b2 = model.paras['b2']
        )


    def save_best_model(self):
        def copy_dir(src, dst):
            os.makedirs(dst, exist_ok=True)
            for filename in os.listdir(src):
                src_path = os.path.join(src, filename)
                dst_path = os.path.join(dst, filename)
                if os.path.isfile(src_path):
                    with open(src_path, 'rb') as f_src:
                        with open(dst_path, 'wb') as f_dst:
                            f_dst.write(f_src.read())
                elif os.path.isdir(src_path):
                    copy_dir(src_path, dst_path)

        def remove_dir(path):
            if not os.path.exists(path):
                return
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    remove_dir(file_path)
            os.rmdir(path)

        current_path = self.save_path
        current_acc = self.acc
        best_path = os.path.join(self.base_dir, 'bestmodel')
        best_txt = os.path.join(best_path, 'train_log.txt')

        if not os.path.exists(best_path) or not os.listdir(best_path):
            print("No bestmodel found. Creating...")
            copy_dir(current_path, best_path)
            return 
        else:
            best_acc = None
            with open(best_txt, 'r') as f:
                for line in f:
                    if line.startswith("Final accuracy"):
                        best_acc = float(line.split(":")[1].strip())
                        break
            if best_acc is None:
                raise ValueError("Best accuracy not found in txt")
            
            print(f"Current Acc: {current_acc:.4f} | Best Acc: {best_acc:.4f}")
            if current_acc > best_acc:
                print("Updating bestmodel...")
                remove_dir(best_path)
                copy_dir(current_path, best_path)
            else:
                print("Keep existing bestmodel.")


    def run(self):
        iL = idxLoader()
        train_datas = (iL.train_img_loader(), iL.train_lbl_loader())
        test_datas = (iL.test_img_loader(), iL.test_lbl_loader())

        Din = train_datas[0].shape[1]
        Dout = max(train_datas[1]) + 1

        if self.extract_path == None:
            model = ThreeLayerNN(Din=Din, H=self.hidden, Dout=Dout)
            record_step = 2
        
            trainer = Trainer(
                model, 
                train_datas, 
                test_datas, 
                self.paras, 
                record_step, 
                self.activation,
                self.save_path,
                self.vis_choice[0]
            )
            trainer.train()

        else:
            weights = np.load(self.extract_path, allow_pickle=True)
            H = weights['w1'].shape[1]
            model = ThreeLayerNN(Din=28*28, H=H, Dout=10)
            model.paras['w1'] = weights['w1']
            model.paras['b1'] = weights['b1']
            model.paras['w2'] = weights['w2']
            model.paras['b2'] = weights['b2']

        if self.vis_choice[1]:
            model.visual_weights1(self.save_path)

        evaluator = Evaluator(
            model, 
            test_datas, 
            self.activation, 
            self.save_path,
            self.vis_choice[2]
            )
        self.acc = evaluator.evaluate()

        if self.save_path:
            self.save_model(model)
            self.save_best_model()
        else:
            print("Model not saved. Set save=True and origin=None to enable saving.")

        return self.acc, model
