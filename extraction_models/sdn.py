import torch
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CNNExtractedMSVD
from constant import *
from helper import *


class SDN(torch.nn.Module):
    def __init__(self, cnn_features_size: int, num_tags: int, dropout_rate: float = 0.3):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(cnn_features_size, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, num_tags),
            torch.nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.classifier(X)


def sdn_loss(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-20
    loss = (truth * torch.log(pred + epsilon) + (1 - truth) * torch.log(1 - pred + epsilon)) * -1
    loss = loss * mask
    loss = torch.sum(loss, dim=1)
    loss = torch.mean(loss, dim=0)
    return loss


def get_accuracy(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    truth = truth.type(torch.LongTensor)
    pred = (pred >= 0.5).type(torch.LongTensor)
    acc = (truth == pred).type(torch.FloatTensor)
    return torch.mean(acc)


if __name__ == '__main__':
    annotation_path = 'annotations.txt'
    train_path = '../extracted/train_val'
    val_path = '../extracted/test'
    cnn_2d_model = 'regnety32'  # regnety32, regnetx32, vgg
    cnn_3d_model = 'resnext101'  # resnext101, shufflenet, shufflenetv2
    batch_size = 20
    epoch = 100
    learning_rate = 5e-3
    ckpt_dir = './checkpoints/sdn'

    # prepare train and validation dataset
    train_dataset = CNNExtractedMSVD(annotation_path, train_path, 300, cnn_2d_model, cnn_3d_model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    train_dataloader_len = len(train_dataloader)

    val_dataset = CNNExtractedMSVD(annotation_path, val_path, 300, cnn_2d_model, cnn_3d_model)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader_len = len(val_dataloader)

    # create and prepare model
    model = SDN(
        cnn_features_size=CNN_3D_FEATURES_SIZE[cnn_3d_model] + CNN_2D_FEATURES_SIZE[cnn_2d_model],
        num_tags=len(train_dataset.tag_dict),
        dropout_rate=0.6,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7, verbose=True)
    loss_func = sdn_loss

    best_acc = 0

    for epoch_idx in range(epoch):
        print(f'\n######### Epoch-{epoch_idx+1} #########')
        train_batch_losses = torch.zeros(train_dataloader_len).to(DEVICE)
        val_batch_losses = torch.zeros(val_dataloader_len).to(DEVICE)
        train_batch_accuracy = torch.zeros(train_dataloader_len).to(DEVICE)
        val_batch_accuracy = torch.zeros(val_dataloader_len).to(DEVICE)

        print('###Training Phase###')
        model.train()
        for batch_idx, (X, y, mask) in enumerate(train_dataloader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)
            out = model(X)

            batch_loss = loss_func(out, y, mask)
            accuracy = get_accuracy(out, y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_batch_losses[batch_idx] = batch_loss.item()
            train_batch_accuracy[batch_idx] = accuracy.item()
            print_batch_loss(batch_loss.item(), batch_idx + 1, train_dataloader_len)

        print('\n###Validation Phase###')
        model.eval()
        for batch_idx, (X, y, mask) in enumerate(val_dataloader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)
            out = model(X)

            batch_loss = loss_func(out, y, mask)
            accuracy = get_accuracy(out, y)

            val_batch_losses[batch_idx] = batch_loss.item()
            val_batch_accuracy[batch_idx] = accuracy.item()
            print_batch_loss(batch_loss.item(), batch_idx + 1, val_dataloader_len)

        avg_train_loss = torch.mean(train_batch_losses).item()
        avg_val_loss = torch.mean(val_batch_losses).item()
        avg_train_accuracy = torch.mean(train_batch_accuracy).item()
        avg_val_accuracy = torch.mean(val_batch_accuracy).item()

        print(f'\nTrain Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}')
        print(f'Train Accuracy: {avg_train_accuracy:.5f}, Validation Accuracy: {avg_val_accuracy:.5f}')
        lr_scheduler.step(avg_val_loss)

        if avg_val_accuracy > best_acc:
            best_acc = avg_val_accuracy

            filename = f'{cnn_2d_model}_{cnn_3d_model}_best.pth'
            filepath = os.path.join(ckpt_dir, filename)
            checkpoint = {
                'cnn_2d_model': cnn_2d_model,
                'cnn_3d_model': cnn_3d_model,
                'model_state_dict': model.state_dict(),
            }
            torch.save(checkpoint, os.path.join(ckpt_dir, filename))
            print(f'Model saved to {filepath}')