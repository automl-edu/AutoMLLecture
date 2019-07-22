import os
import argparse
import logging
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary

from cnn import torchModel
from datasets import K49


def main(model_config,
         data_dir,
         num_epochs=10,
         batch_size=50,
         learning_rate=0.001,
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer=torch.optim.Adam,
         data_augmentations=None,
         save_model_str=None):
    """
    Training loop for configurableNet.
    :param model_config: network config (dict)
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during trainnig (torch.optim.Optimizer)
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :return:
    """

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if data_augmentations is None:
        # You can add any preprocessing/data augmentation you want here
        data_augmentations = transforms.ToTensor()
    elif isinstance(type(data_augmentations), list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    train_dataset = K49(data_dir, True, data_augmentations)
    test_dataset = K49(data_dir, False, data_augmentations)

    # Make data batch iterable
    # Could modify the sampler to not uniformly random sample
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    model = torchModel(model_config,
                       input_shape=(
                           train_dataset.channels,
                           train_dataset.img_rows,
                           train_dataset.img_cols
                       ),
                       num_classes=train_dataset.n_classes
            ).to(device)
    total_model_params = np.sum(p.numel() for p in model.parameters())
    # instantiate optimizer
    optimizer = model_optimizer(model.parameters(),
                                lr=learning_rate)
    # instantiate training criterion
    train_criterion = train_criterion().to(device)

    logging.info('Generated Network:')
    summary(model, (train_dataset.channels,
                    train_dataset.img_rows,
                    train_dataset.img_cols),
            device='cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        train_score, train_loss = model.train_fn(optimizer, train_criterion,
                                              train_loader, device)
        logging.info('Train accuracy %f', train_score)

        test_score = model.eval_fn(test_loader, device)
        logging.info('Test accuracy %f', test_score)

    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        if os.path.exists(save_model_str):
            save_model_str += '_'.join(time.ctime())
        torch.save(model.state_dict(), save_model_str)


if __name__ == '__main__':
    """
    This is just an example of how you can use train and evaluate
    to interact with the configurable network
    """
    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss,
                 'mse': torch.nn.MSELoss}
    opti_dict = {'adam': torch.optim.Adam,
                 'adad': torch.optim.Adadelta,
                 'sgd': torch.optim.SGD}

    cmdline_parser = argparse.ArgumentParser('AutoML SS19 final project')

    cmdline_parser.add_argument('-e', '--epochs',
                                default=10,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=96,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default='../data',
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-l', '--learning_rate',
                                default=0.01,
                                help='Optimizer learning rate',
                                type=float)
    cmdline_parser.add_argument('-L', '--training_loss',
                                default='cross_entropy',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='adam',
                                help='Which optimizer to use during training',
                                choices=list(opti_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-m', '--model_path',
                                default=None,
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    # architecture parametrization
    architecture = {
            'n_layers': 1,
        }

    main(
        architecture,
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=opti_dict[args.optimizer],
        data_augmentations=None,  # Not set in this example
        save_model_str=args.model_path
    )
