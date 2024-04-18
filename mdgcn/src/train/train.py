import sys
import os
sys.path.append(os.path.abspath('../..'))
import torch
from torch import nn
from src.mdgcn.mdgcn import MDGCN
from src.dataloader import dataloader, fsample
from src.dataloader.dataloader import Dataloader
from src.train.metrics import Metric, PlotTool
from src.train import outstream
from src.train.outstream import tprint
import random
import numpy as np
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


# train and test configurations
class Config:
    def __init__(self, dataset='kickstarter', weights_name=None):
        assert dataset in ['kickstarter', 'indiegogo']
        self.test_mode = False
        self.dataset = dataset
        fsample.DATASET = dataset
        self.weights_name = weights_name
        self.out_dir = os.path.abspath(f"../../out/{self.dataset}_out")
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        outstream.output_file_path = f"{self.out_dir}/out.txt"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # hyper parameters
        self.epochs = 80 if dataset == 'kickstarter' else 45
        self.batch_size = 8
        self.optim_method = 'Adam'
        self.lr = 5e-5
        self.wd = 0

    def weights_path(self):
        dir_path = f"{self.out_dir}/mdgcn"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path = "{}/{}".format(dir_path, self.weights_name)
        return path

    def __str__(self):
        res = "dataset--{}, device--{}, optim_method--{}, \n".format(
            self.dataset, self.device, self.optim_method
        )
        res += "batch_size--{}, lr--{:.3e}, wd--{:.3e}".format(
            self.batch_size, self.lr, self.wd
        )
        return res


class TrainTool:
    def __construct_for_train(self):
        optim_method = self.cfg.optim_method.lower()
        self.optimizer = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop
        }[optim_method](self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd)
        self.train_dataloader = Dataloader(
            dataset_type=dataloader.TRAIN, batch_size=self.cfg.batch_size,
            device=self.cfg.device
        )
        self.val_dataloader = Dataloader(
            dataset_type=dataloader.VAL, batch_size=1,
            device=self.cfg.device
        )
        self.train_num = len(self.train_dataloader)
        self.val_num = len(self.val_dataloader)

    def __construct_for_test(self):
        self.test_dataloader = Dataloader(
            dataset_type=dataloader.TEST, batch_size=1,
            device=self.cfg.device
        )
        self.test_num = len(self.test_dataloader)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = MDGCN(
            channels=512,
            conv_times_list=[1, 2, 3, 4],
            inner_k_list=[11, 9, 7, 5],
            cross_k_list=[13, 11, 9, 7],
            drop_out=0.5,
            drop_path=0.5
        )
        if self.cfg.weights_name is not None:
            weights = torch.load(self.cfg.weights_path())
            self.model.load_state_dict(weights)
        self.model = self.model.to(self.cfg.device)
        self.loss_func = nn.CrossEntropyLoss().to(self.cfg.device)
        if self.cfg.test_mode:
            self.__construct_for_test()
            tprint("test_infos: dataset--{}, test_num--{}, device--{}".format(
                cfg.dataset, self.test_num, self.cfg.device), display_time=True
            )
        else:
            self.__construct_for_train()
            tprint("train_infos: ", display_time=True)
            tprint("{}".format(self.cfg))
            tprint("train_num--{}, val_num--{}".format(self.train_num, self.val_num))
            tprint("\n\n")

    def eval(self, val_dataloader=None):
        if val_dataloader is None:
            val_dataloader = self.val_dataloader
        pred = torch.zeros((0, 2), dtype=torch.float32)
        true_labels = torch.zeros(0, dtype=torch.float32)
        loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch_inputs, batch_labels in val_dataloader:
                batch_pred = self.model(batch_inputs)
                batch_loss = self.loss_func(batch_pred, batch_labels)
                loss += batch_loss * batch_labels.shape[0]
                pred = torch.cat((pred, batch_pred.cpu()), dim=0)
                true_labels = torch.cat((true_labels, batch_labels.cpu()), dim=0)
        loss = loss / len(val_dataloader)
        res = Metric(pred, true_labels, loss)
        return res

    def train_step(self):
        pred = torch.zeros((0, 2), dtype=torch.float32)
        true_labels = torch.zeros(0, dtype=torch.float32)
        loss = 0.0
        self.model.train()
        for batch_inputs, batch_labels in self.train_dataloader:
            # forward and compute the loss
            batch_pred = self.model(batch_inputs)
            batch_loss = self.loss_func(batch_pred, batch_labels)
            loss += batch_loss * batch_labels.shape[0]
            # save the pred and labels
            pred = torch.cat((pred, batch_pred.cpu()), dim=0)
            true_labels = torch.cat((true_labels, batch_labels.cpu()), dim=0)
            # backward
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        # calculate metrics
        loss = loss / self.train_num
        res = Metric(pred, true_labels, loss)
        return res

    def train(self):
        out_path = f"{self.cfg.out_dir}/metrics.png"
        plot_tool = PlotTool(out_path=out_path)
        # multiple epochs
        for i in range(self.cfg.epochs):
            # train and validate
            train_res = self.train_step()
            val_res = self.eval()
            # print the results
            tprint(f"[{i + 1}/{self.cfg.epochs}]", display_time=True)
            tprint(f"train: {train_res}")
            tprint(f"  val: {val_res}")
            tprint("\n\n")
            # plot
            plot_tool.append_and_plot(train_res, val_res)
            # save models
            dir_path = f"{self.cfg.out_dir}/mdgcn"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            weights_name = f"epoch({i + 1})_" + val_res.generate_fileName() + ".pth"
            weights_path = f"{dir_path}/{weights_name}"
            torch.save(self.model.state_dict(), weights_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kickstarter')
    args = parser.parse_args()
    cfg = Config(dataset=args.dataset)
    tt = TrainTool(cfg)
    tt.train()


if __name__ == '__main__':
    main()


