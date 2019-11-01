import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from utils import AvgrageMeter, accuracy


class torchModel(nn.Module):
    def __init__(self, config, input_shape=(1, 28, 28), num_classes=10):
        super(torchModel, self).__init__()
        layers = []
        n_layers = config['n_layers']
        n_conv_layers = 1
        kernel_size = 2
        in_channels = input_shape[0]
        out_channels = 4

        for i in range(n_conv_layers):
            c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=2, padding=1
                         )
            a = nn.ReLU(inplace=False)
            p = nn.MaxPool2d(kernel_size=2, stride=1)
            layers.extend([c, a, p])
            in_channels = out_channels
            out_channels *= 2

        self.conv_layers = nn.Sequential(*layers)
        self.output_size = num_classes

        self.fc_layers = nn.ModuleList()
        n_in = self._get_conv_output(input_shape)
        n_out = 256
        for i in range(n_layers):
            fc = nn.Linear(int(n_in), int(n_out))
            self.fc_layers += [fc]
            n_in = n_out
            n_out /= 2

        self.last_fc = nn.Linear(int(n_in), self.output_size)
        self.dropout = nn.Dropout(p=0.2)

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers:
            x = self.dropout(F.relu(fc_layer(x)))
        x = self.last_fc(x)
        return x

    def train_fn(self, optimizer, criterion, loader, device, train=True):
        """
        Training method
        :param optimizer: optimization algorithm
        :criterion: loss function
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: (accuracy, loss) on the data
        """
        score = AvgrageMeter()
        objs = AvgrageMeter()
        self.train()

        t = tqdm(loader)
        for images, labels in t:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            acc, _ = accuracy(logits, labels, topk=(1, 5))
            n = images.size(0)
            objs.update(loss.item(), n)
            score.update(acc.item(), n)

            t.set_description('(=> Training) Loss: {:.4f}'.format(objs.avg))

        return score.avg, objs.avg

    def eval_fn(self, loader, device, train=False):
        """
        Evaluation method
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: accuracy on the data
        """
        score = AvgrageMeter()
        self.eval()

        t = tqdm(loader)
        with torch.no_grad():  # no gradient needed
            for images, labels in t:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                acc, _ = accuracy(outputs, labels, topk=(1, 5))
                score.update(acc.item(), images.size(0))

                t.set_description('(=> Test) Score: {:.4f}'.format(score.avg))

        return score.avg

