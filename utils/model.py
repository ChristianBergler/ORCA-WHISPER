import torch.nn as nn


def weights_init_kaiming(model):
    if type(model) == nn.Conv2d or type(model) == nn.Conv1d:
        nn.init.kaiming_normal_(model.weight, nonlinearity='leaky_relu')
    elif type(model) == nn.ConvTranspose2d or type(model) == nn.ConvTranspose1d:
        nn.init.kaiming_normal_(model.weight, nonlinearity='relu')
    elif type(model) == nn.Linear:
        model.weight.data.normal_(0.0, 0.002)
        model.bias.data.fill_(0)


def weights_init_normal(model):
    if type(model) == nn.Conv2d or type(model) == nn.Conv1d:
        model.weight.data.normal_(0.0, 0.02)
    elif type(model) == nn.BatchNorm2d:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def weights_init_xavier(model):
    if type(model) == nn.Conv2d or type(model) == nn.Conv1d:
        nn.init.xavier_normal_(model.weight)
    elif type(model) == nn.ConvTranspose2d or type(model) == nn.ConvTranspose1d:
        nn.init.xavier_normal_(model.weight)
    elif type(model) == nn.Linear:
        nn.init.xavier_normal_(model.weight)
        model.bias.data.fill_(0)

