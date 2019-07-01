import os
import sys
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from model_search import Network


class Architect(object):
    """
    Base class for the architect, which is just an optimizer (different from
    the one used to update the model parameters) for the architectural parameters.
    """
    def __init__(self, model):
        """
        :model: nn.Module; search model
        """
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.arch_optimizer = torch.optim.Adam(
                self.model.arch_parameters,
                lr=3e-4,
                betas=(0.5, 0.999),
                weight_decay=1e-3
        )

    def step(self, input_valid, target_valid):
        """
        This method computes a gradient step in the architecture space, i.e.
        updates the self.model.alphas_normal and self.model.alphas_reduce by
        the gradient of the validation loss with respect to these alpha
        parameters.
        :input_valid: torch.Tensor; validation mini-batch
        :target_valid: torch.Tensor: ground truth labels of this mini-batch
        """
        self.arch_optimizer.zero_grad()
        #TODO: do a forward pass using the validation mini-batch input
        logits = self.model(input_valid)
        #TODO: compute the loss using self.criterion and backpropagate to
        #      compute the gradients w.r.t. the alphas
        loss = self.criterion(logits, target_valid)
        loss.backward()
        #TODO: do a step in the architecture space using the
        #      self.arch_optimizer
        self.arch_optimizer.step()


def train(train_loader, valid_loader, model, architect, criterion,
          optimizer, device):
    """
    Training loop. This function computes the DARTS loop, i.e. it takes one
    step in the architecture space and one in the weight space in an
    interleaving manner. For the architectural updates we use the validation
    set and for the search model parameters updates the training set. In DARTS
    these two sets have equal sizes, which in the case of MNIST it is 30k
    examples per each.
    """
    objs = utils.AvgrageMeter()
    accr = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)

        # get a random minibatch from the search queue with replacement and
        # update the architectural parameters with the validation loss gradient
        # NOTE: The model parameters are kept fixed here, just the alphas are
        # updated
        input_search, target_search = next(iter(valid_loader))
        architect.step(input_search.to(device), target_search.to(device))

        # update the search model parameters with the updated architecture
        # NOTE: The architecture is kept fixed here, just the search model
        # weights/parameters are updated
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        _accr = utils.accuracy(logits, target)
        objs.update(loss.item(), input.size(0))
        accr.update(_accr.item(), input.size(0))

        logging.info('train mini-batch %03d, loss=%e accuracy=%f', step,
                     objs.avg, accr.avg)

    return accr.avg, objs.avg


def infer(valid_loader, model, criterion, device):
    """
    Compute the accuracy on the validation set (the same used for updating the
    architecture).
    """
    objs = utils.AvgrageMeter()
    accr = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_loader):
            input, target = input.to(device), target.to(device)

            logits = model(input)
            loss = criterion(logits, target)

            _accr= utils.accuracy(logits, target)
            objs.update(loss.item(), input.size(0))
            accr.update(_accr.item(), input.size(0))

            logging.info('valid mini-batch %03d, loss=%e accuracy=%f', step,
                         objs.avg, accr.avg)

    return accr.avg, objs.avg


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s", args)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader, valid_loader = utils.search_dataloader(args, kwargs)

    criterion = nn.CrossEntropyLoss().to(device)
    model = Network(device, nodes=2).to(device)

    logging.info("param size = %fMB",
            np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6)

    optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay)

    architect = Architect(model)

    for epoch in range(args.epochs):
        logging.info("Starting epoch %d/%d", epoch+1, args.epochs)

        # training
        train_acc, train_obj = train(train_loader, valid_loader, model,
                architect, criterion, optimizer, device)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_loader, model, criterion, device)
        logging.info('valid_acc %f', valid_acc)

        # compute the discrete architecture from the current alphas
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        with open(args.save + '/architecture', 'w') as f:
            f.write(str(genotype))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("mnist")
    parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='num of training epochs')
    parser.add_argument('--save', type=str, default='logs', help='path to logs')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    args = parser.parse_args()

    # logging utilities
    os.makedirs(args.save, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main(args)

