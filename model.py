import numpy as np
import matplotlib.pyplot as plt
import os

class ThreeLayerNN:
    def __init__(self, Din, H, Dout):
        self.paras = {}
        self.paras['w1'] = np.random.randn(Din, H) * np.sqrt(1.0 / Din)
        self.paras['b1'] = np.zeros(H)
        self.paras['w2'] = np.random.randn(H, Dout) * np.sqrt(1.0 / H)
        self.paras['b2'] = np.zeros(Dout)

    def relu(self, z):
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))
    
    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True) # avoid overflow
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def cross_entropy(self, y_pred, y):
        eps = 1e-15
        return -np.sum(y * np.log(y_pred + eps)) / y.shape[0]
    
    def loss(self, y_pred, y, reg = 0.0):
        data_loss = self.cross_entropy(y_pred, y)
        w1, w2 = self.paras['w1'], self.paras['w2']
        reg_loss = reg * (np.sum(w1 ** 2) + np.sum(w2 ** 2))
        return data_loss + reg_loss

    def forward_sigmoid(self, x):
        w1, b1 = self.paras['w1'], self.paras['b1']
        w2, b2 = self.paras['w2'], self.paras['b2']
        z1 = x.dot(w1) + b1
        h = self.sigmoid(z1)
        y_pred = self.softmax(h.dot(w2) + b2)
        return y_pred, h, z1
    
    def backward_sigmoid(self, x, h, z1, y_pred, y, lr, reg = 0.0):
        N = x.shape[0]
        grads = {}
        w1, w2 = self.paras['w1'], self.paras['w2']
        dz2 = (y_pred - y) / N
        grads['w2'] = h.T.dot(dz2) + 2 * reg * w2
        grads['b2'] = dz2.sum(axis=0)
        dh = dz2.dot(w2.T)
        grads['w1'] = x.T.dot(dh * h * (1 - h)) + 2 * reg * w1
        grads['b1'] = (dh * h * (1 - h)).sum(axis=0)

        self.paras['w1'] -= lr * grads['w1']
        self.paras['b1'] -= lr * grads['b1']
        self.paras['w2'] -= lr * grads['w2']
        self.paras['b2'] -= lr * grads['b2']
        return grads
    
    def forward_relu(self, x):
        w1, b1 = self.paras['w1'], self.paras['b1']
        w2, b2 = self.paras['w2'], self.paras['b2']

        z1 = x.dot(w1) + b1
        h = self.relu(z1)

        z2 = h.dot(w2) + b2
        y_pred = self.softmax(z2)

        return y_pred, h, z1
    
    def backward_relu(self, x, h, z1, y_pred, y, lr, reg = 0.0):
        N = x.shape[0]
        grads = {}

        w1, w2 = self.paras['w1'], self.paras['w2']
        dz2 = (y_pred - y) / N
        grads['w2'] = h.T.dot(dz2) + 2 * reg * w2
        grads['b2'] = dz2.sum(axis=0)

        dh = dz2.dot(w2.T)
        dz1 = dh * (z1 > 0)
        grads['w1'] = x.T.dot(dz1) + 2 * reg * w1
        grads['b1'] = dz1.sum(axis=0)

        self.paras['w1'] -= lr * grads['w1']
        self.paras['b1'] -= lr * grads['b1']
        self.paras['w2'] -= lr * grads['w2']
        self.paras['b2'] -= lr * grads['b2']
        return grads
    
    
    def visual_weights1(self, save_path):
        ws = self.paras['w1']
        num = ws.shape[1]
        if num > 100:
            print(f"Too many neurons ({num}) to visualize, showing only the first 100.")
            ws = ws[:, :100]
            num = 100

        cols = int(np.ceil(np.sqrt(num)))
        rows = int(np.ceil(num / cols))

        plt.figure(figsize=(cols, rows))  # 自适应大小
        plt.suptitle('Visualization of w1 Weights')

        for i in range(num):
            w = ws[:,i]
            w = w.reshape(28, 28)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(w, cmap='gray')
            plt.title(f'Neuron {i+1}', fontsize=6)
            plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'w1_visualize.png'))
        plt.close()

