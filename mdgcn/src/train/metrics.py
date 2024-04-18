import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


class Metric:
    def __init__(self, pred, true_labels, loss):
        threshold = 0.5
        self.pred_probas = torch.softmax(pred, dim=-1)[:, 1].cpu().detach().numpy()
        self.pred_labels = (self.pred_probas > threshold).astype(np.int8)
        self.true_labels = true_labels.cpu().detach().numpy().astype(np.int8)
        self.loss = loss.cpu().detach().numpy()
        self.accuracy = accuracy_score(self.true_labels, self.pred_labels)
        self.precision = precision_score(self.true_labels, self.pred_labels)
        self.recall = recall_score(self.true_labels, self.pred_labels)
        self.f1 = f1_score(self.true_labels, self.pred_labels)
        self.auc = roc_auc_score(self.true_labels, self.pred_probas)

    def __str__(self):
        res = (
            "loss--{:.4f}, "
            "accuracy--{:.4f}, "
            "precision--{}, "
            "recall--{}, "
            "f1--{}, "
            "auc--{}"
        ).format(
            self.loss, self.accuracy,
            np.round(self.precision, 4),
            np.round(self.recall, 4),
            np.round(self.f1, 4),
            np.round(self.auc, 4)
        )
        return res

    def generate_fileName(self):
        file_name = "loss({:.4f})_accuracy({:.4f})_f1({:.4f})".format(
            self.loss, self.accuracy, self.f1
        )
        return file_name


class PlotTool:
    def __init__(self, out_path='metrics.png'):
        self.title = 'mdgcn'
        self.out_path = out_path
        # loss
        self.train_loss_list = []
        self.val_loss_list = []
        # accuracy
        self.train_accuracy_list = []
        self.val_accuracy_list = []
        # f1
        self.train_f1_list = []
        self.val_f1_list = []

    # plot
    @staticmethod
    def __plot_curves(loc, x, y_list, x_label_name, y_label_name, title=None):
        curve_names = ['train', 'val']
        markers = ['+', '^']
        plt.subplot(1, 3, loc)
        if title is not None:
            plt.title(title)
        plt.xlabel(x_label_name)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel(y_label_name)
        plt.grid(True)
        for i in range(len(curve_names)):
            y = y_list[i]
            if x.shape[0] < 16:
                plt.plot(x, y, label=curve_names[i], linewidth='0.8', marker=markers[i])
            else:
                plt.plot(x, y, label=curve_names[i], linewidth='0.8')
            plt.legend(loc='best', frameon=False)

    # accept a new result and update the figures
    def append_and_plot(self, train_res: Metric, val_res: Metric):
        # train res
        self.train_loss_list.append(train_res.loss)
        self.train_accuracy_list.append(train_res.accuracy)
        self.train_f1_list.append(np.mean(train_res.f1))
        # val res
        self.val_loss_list.append(val_res.loss)
        self.val_accuracy_list.append(val_res.accuracy)
        self.val_f1_list.append(np.mean(val_res.f1))
        # plot
        plt.figure(figsize=(15.0, 5.0))
        # loss-epoch
        x = np.arange(1, 1 + len(self.train_loss_list))
        y_list = [np.array(self.train_loss_list), np.array(self.val_loss_list)]
        PlotTool.__plot_curves(loc=1, x=x, y_list=y_list,
                               x_label_name='epoch', y_label_name='loss', title=self.title)
        # accuracy-epoch
        y_list = [np.array(self.train_accuracy_list), np.array(self.val_accuracy_list)]
        PlotTool.__plot_curves(loc=2, x=x, y_list=y_list,
                               x_label_name='epoch', y_label_name='accuracy')
        # f1-epoch
        y_list = [np.array(self.train_f1_list), np.array(self.val_f1_list)]
        PlotTool.__plot_curves(loc=3, x=x, y_list=y_list,
                               x_label_name='epoch', y_label_name='f1')
        # save the figure
        plt.savefig(self.out_path)


