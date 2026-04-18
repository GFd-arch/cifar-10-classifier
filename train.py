import numpy as np
import math
import os
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, trains, preds, paras, step = 1, activation = 'relu', save_path = None, vis=True):
        self.model = model
        self.trains = trains
        self.preds = preds
        self.paras = paras
        self.activation = activation
        self.save_path = save_path
        self.vis = vis

        self.train_loss = []
        self.pred_loss = []
        self.train_accs = []
        self.pred_accs = []
        self.record_step = step

    def step_decay(self, e, epoch, step_size=50, decay_rate=0.01):
        steps = e // step_size
        return decay_rate ** steps
    
    def exp_decay(self, e, epoch, decay_rate=0.999):
        return decay_rate ** e
    
    def linear_decay(self, e, epoch, min=0.1):
        progress = e / epoch
        return max(min, 1.0 - progress * (1 - min))
    
    def cosine_decay(self, e, epoch, min=0.1):
        cosine = 0.5 * (1 + math.cos(math.pi * e / epoch))
        return min + (1 - min) * cosine
    

    def visualize(self):
        losses = [self.train_loss, self.pred_loss]
        accs = [self.train_accs, self.pred_accs] if self.train_accs else [self.pred_accs]
        numacc = len(accs)

        plt.figure(figsize=(12, 6))

        train, pred = losses
        x_axis = np.arange(1, len(train) + 1) * self.record_step

        plt.subplot(1,2,1)
        plt.plot(x_axis, train, color='red', linewidth=2, label='Train', alpha=0.8)
        plt.plot(x_axis, pred, color='blue', linewidth=2, label='Validation', alpha=0.8)
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1,2,2)

        if numacc == 1:
            acc = accs[0]
            x_axis_acc = np.arange(1, len(acc) + 1) * self.record_step
            plt.plot(x_axis_acc, acc, color='green', linewidth=2, label='Accuracy', alpha=0.8)

        else:
            train_acc, pred_acc = accs
            x_axis_acc = np.arange(1, len(train_acc) + 1) * self.record_step

            plt.plot(x_axis_acc, train_acc, color='red', linewidth=2, label='Train Accuracy', alpha=0.8)
            plt.plot(x_axis_acc, pred_acc, color='blue', linewidth=2, label='Validation Accuracy', alpha=0.8)

        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, 'train_plot.png'))
        plt.close()


    def save_log(self):
        if self.save_path:
            path = os.path.join(self.save_path, 'train_log.txt')
            with open(path, 'w') as f:
                f.write("Final accuracy: {}\n\n".format(str(self.pred_accs[-1])))
                f.write("Hidden Layer Size: {}\n".format(self.model.paras['w1'].shape[1]))

                f.write("Training Parameters:\n")
                f.write("Epochs: {}\n".format(self.paras[0]))
                f.write("Learning Rate: {}, decay: {}\n".format(self.paras[1], self.paras[4]))
                f.write("Regularization: {}\n".format(self.paras[2]))
                f.write("Batch Size: {}\n\n".format(self.paras[3]))

                f.write("Train Loss:\n")
                f.write(str(self.train_loss[-1]) + "\n\n")

                f.write("Test Loss:\n")
                f.write(str(self.pred_loss[-1]) + "\n\n")

                f.write("Train Acc:\n")
                f.write(str(self.train_accs[-1]) + "\n\n")

                f.write("Test Acc:\n")
                f.write(str(self.pred_accs[-1]) + "\n")


    def train(self):
        epoch, lr, reg, batchsize, decay = self.paras
        train_data, train_label = self.trains
        test_data, test_label = self.preds
        N = train_data.shape[0]
        Dout = np.max(train_label) + 1

        y_test_onehot = np.zeros((len(test_data), Dout))
        y_test_onehot[np.arange(len(test_data)), test_label] = 1

        for e in range(epoch):
            if decay == 'step':
                learning_rate = lr * self.step_decay(e, epoch)
            elif decay == 'exp':
                learning_rate = lr * self.exp_decay(e, epoch)
            elif decay == 'linear':
                learning_rate = lr * self.linear_decay(e, epoch)
            elif decay == 'cosine':
                learning_rate = lr * self.cosine_decay(e, epoch, min = 0.01)
            else:
                learning_rate = lr

            perm = np.random.permutation(N)
            train_data_shuffled = train_data[perm]
            train_label_shuffled = train_label[perm]

            for i in range(0, N, batchsize):
                x = train_data_shuffled[i:i + batchsize]
                y = train_label_shuffled[i:i + batchsize]

                B = x.shape[0]

                y_onehot = np.zeros((B, Dout))
                y_onehot[np.arange(B), y] = 1

                if self.activation == 'relu':
                    y_pred, h, z1 = self.model.forward_relu(x)
                    loss = self.model.loss(y_pred, y_onehot, reg=reg)
                    self.model.backward_relu(x, h, z1, y_pred, y_onehot, lr=learning_rate, reg=reg)
                elif self.activation == 'sigmoid':
                    y_pred, h, z1 = self.model.forward_sigmoid(x)
                    loss = self.model.loss(y_pred, y_onehot, reg=reg)
                    self.model.backward_sigmoid(x, h, z1, y_pred, y_onehot, lr=learning_rate, reg=reg)
                else:
                    raise ValueError(f"Unsupported activation function: {self.activation}")

            if (e + 1) % 10 == 0:
                print(f"epoch: {e+1}, loss: {loss:.6f}")

            if (e + 1) % self.record_step == 0:
                self.train_loss.append(loss)

                if self.activation == 'relu':
                    y_test_pred, _, _ = self.model.forward_relu(test_data)
                else:
                    y_test_pred, _, _ = self.model.forward_sigmoid(test_data)

                self.pred_loss.append(self.model.loss(y_test_pred, y_test_onehot))

                if self.activation == 'relu':
                    y_train_pred_all, _, _ = self.model.forward_relu(train_data)
                else:
                    y_train_pred_all, _, _ = self.model.forward_sigmoid(train_data)

                train_acc = np.mean(np.argmax(y_train_pred_all, axis=1) == train_label)
                self.train_accs.append(train_acc)

                test_acc = np.mean(np.argmax(y_test_pred, axis=1) == test_label)
                self.pred_accs.append(test_acc)

        if self.save_path:
            self.save_log()

        if self.vis:
            self.visualize()
