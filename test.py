import numpy as np
import matplotlib.pyplot as plt
import os

class Evaluator():
    def __init__(self, model, test_datas, activation = 'relu', save_path = None, vis=True):
        self.model = model
        self.test_datas = test_datas
        self.activation = activation
        self.save_path = save_path
        self.vis = vis

    def confusion_matrix(self, preds, labels):
        num_classes = np.max(labels) + 1
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true_label, pred_label in zip(labels, preds):
            conf_matrix[true_label][pred_label] += 1
        conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)


        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, cmap='Reds')
        plt.colorbar()

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        plt.xticks(np.arange(10))
        plt.yticks(np.arange(10))

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, conf_matrix[i, j],
                        ha="center", va="center",
                        color="blue", fontsize=8)

        if self.save_path:
            plt.savefig(os.path.join(self.save_path, 'confusion_matrix.png'))
        plt.close()


    def evaluate(self):
        test_data, test_label = self.test_datas
        x, y = test_data, test_label

        if self.activation == 'relu':
            y_pred, _, _ = self.model.forward_relu(x)
        elif self.activation == 'sigmoid':
            y_pred, _, _ = self.model.forward_sigmoid(x)
        else:
            raise ValueError("Unsupported activation function: {}".format(self.activation))
        
        correct = np.sum(np.argmax(y_pred, axis=1) == y)
        total = len(y)

        if self.vis:
            self.confusion_matrix(np.argmax(y_pred, axis=1), y)

        return correct / total