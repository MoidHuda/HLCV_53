from functools import partial

import torch
import torch.nn as nn

from src.data_loaders.data_modules import CIFAR10DataModule
from src.trainers.cnn_trainer import CNNTrainer
from src.models.cnn.model import ConvNet
from src.models.cnn.metric import TopKAccuracy
from copy import deepcopy

q1_experiment = dict(
    name = 'CIFAR10_CNN',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = None,
        drop_prob = 0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 10,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "off",
        early_stop = 0,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),

)


#########  TODO #####################################################
#  You would need to create the following config dictionaries       #
#  to use them for different parts of Q2 and Q3.                    #
#  Feel free to define more config files and dictionaries if needed.#
#  But make sure you have a separate config for every question so   #
#  that we can use them for grading the assignment.                 #
#####################################################################
q2a_normalization_experiment = dict(
    name = 'CIFAR10_CNN_Normalization',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm2d,
        drop_prob = 0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 10,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "off",
        early_stop = 0,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),
)

q2b_1a_epoch_experiment = deepcopy(q1_experiment)
q2b_1a_epoch_experiment["name"] = "CIFAR10_CNN_1a_Epoch50"
q2b_1a_epoch_experiment["trainer_config"]["monitor"] = "max eval_top1"
q2b_1a_epoch_experiment["trainer_config"]["epochs"] = 50

q2b_2a_epoch_experiment = deepcopy(q2a_normalization_experiment)
q2b_2a_epoch_experiment["name"] = "CIFAR10_CNN_2a_Epoch50"
q2b_2a_epoch_experiment["trainer_config"]["monitor"] = "max eval_top1"
q2b_2a_epoch_experiment["trainer_config"]["epochs"] = 50

q2c_earlystop_experiment = deepcopy(q2b_2a_epoch_experiment)
q2c_earlystop_experiment["name"] = "CIFAR10_CNN_2c_EarlyStop"
q2c_earlystop_experiment["trainer_config"]["early_stop"] = 4

q3a_aug1_experiment = deepcopy(q2b_2a_epoch_experiment)
q3a_aug1_experiment["name"] = "CIFAR10_CNN_3a_Aug1"
q3a_aug1_experiment["trainer_config"]["epochs"] = 20
q3a_aug1_experiment["data_args"]["transform_preset"] = "CIFAR10_WithFlip"

q3a_aug2_experiment = deepcopy(q3a_aug1_experiment)
q3a_aug2_experiment["name"] = "CIFAR10_CNN_3a_Aug2"
q3a_aug2_experiment["data_args"]["transform_preset"] = "CIFAR10_WithCrop"

q3a_aug3_experiment = deepcopy(q3a_aug1_experiment)
q3a_aug3_experiment["name"] = "CIFAR10_CNN_3a_Aug3"
q3a_aug3_experiment["data_args"]["transform_preset"] = "CIFAR10_WithFlipAndCrop"

q3a_aug4_experiment = deepcopy(q3a_aug1_experiment)
q3a_aug4_experiment["name"] = "CIFAR10_CNN_3a_Aug4"
q3a_aug4_experiment["data_args"]["transform_preset"] = "CIFAR10_WithColorJitter"

q3a_aug5_experiment = deepcopy(q3a_aug1_experiment)
q3a_aug5_experiment["name"] = "CIFAR10_CNN_3a_Aug5"
q3a_aug5_experiment["data_args"]["transform_preset"] = "CIFAR10_WithGrayScale"

q3a_aug6_experiment = deepcopy(q3a_aug1_experiment)
q3a_aug6_experiment["name"] = "CIFAR10_CNN_3a_Aug6"
q3a_aug6_experiment["data_args"]["transform_preset"] = "CIFAR10_WithFlipAndGrayScale"

q3a_aug7_experiment = deepcopy(q3a_aug1_experiment)
q3a_aug7_experiment["name"] = "CIFAR10_CNN_3a_Aug7"
q3a_aug7_experiment["trainer_config"]["epochs"] = 30
q3a_aug7_experiment["data_args"]["transform_preset"] = "CIFAR10_WithFlipAndGrayScaleAndColorJitter"



q3b_dropout_experiment = ()

# define more config dictionaries if needed...