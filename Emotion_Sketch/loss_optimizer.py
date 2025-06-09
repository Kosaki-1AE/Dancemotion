# loss_optimizer.py　損失関数
import torch
import torch.nn as nn
import torch.optim as optim


def get_loss_and_optimizer(model, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    return criterion, optimizer
