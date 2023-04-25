from io import TextIOWrapper
from typing import List
from constant import EOS_TAG
import argparse


def print_batch_loss(loss_value: float, current_batch: int, total_batch: int):
    print(' ' * 80, end='\r')
    print(f'Step [{current_batch}/{total_batch}] Current Loss: {loss_value:.5f}', end='\r')


def print_test_overfit(epoch_loss_value: float, current_epoch: int, total_epoch: int):
    print(' ' * 80, end='\r')
    print(f'Epoch [{current_epoch}/{total_epoch}] Current Loss: {epoch_loss_value:.5f}', end='\r')


def create_batch_log_file(filepath: str) -> TextIOWrapper:
    file = open(filepath, 'w')
    file.write('Index, Training Loss\n')
    return file


def create_epoch_log_file(filepath: str) -> TextIOWrapper:
    file = open(filepath, 'w')
    file.write('Index, Training Loss, Validation Loss\n')
    return file

def format_result(result: List) -> str:
    try:
        eos_idx = result.index(EOS_TAG)
        formatted = result[:eos_idx]
        return ' '.join(formatted)
    except:
        return ' '.join(result)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--annotation-path', help='File path to annotation', required=True)
    parser.add_argument('--dataset-dir', help='Directory path to dataset for train and validation', required=True)
    parser.add_argument('--ckpt-dir', help='Checkpoint directory, will save for each epoch', default='./checkpoints')
    parser.add_argument('--ckpt-interval', help='How many epoch between checkpoints', default=1, type=int)
    parser.add_argument('--log-dir', help='Log directory', default='./logs')
    parser.add_argument('--timestep', help='Total timestep', default=80, type=int)
    parser.add_argument('--batch-size', help='Batch size for training', default=8, type=int)
    parser.add_argument('--epoch', help='Total epoch', default=20, type=int)
    parser.add_argument('--learning-rate', help='Learning rate for training', default=1e-4, type=float)
    parser.add_argument('--model-path', help='Load pretrained model')
    parser.add_argument(
        '--model-cnn-2d',
        help='2D CNN model architecture',
        choices=['vgg', 'regnetx32', 'regnety32', 'resnext'],
        required=True,
    )
    parser.add_argument(
        '--model-cnn-3d',
        help='3D CNN model architecture',
        choices=['shufflenet', 'shufflenetv2', 'resnext101', 'eco'],
        required=True,
    )
    parser.add_argument(
        '--test-overfit',
        help='Sanity check to test overfit model with very small dataset',
        action='store_true',
    )
    parser.add_argument(
        '--mode',
        help='Use sample distribution or argmax',
        choices=['sample', 'argmax'],
        required=True,
    )

    return parser.parse_args()

def print_training_config(args):
    print('Training configuration:')
    print('Annotation file:', args.annotation_path)
    print('2D CNN model:', args.model_cnn_2d)
    print('3D CNN model:', args.model_cnn_3d)
    print('Generate mode:', args.mode)
    print('Dataset directory:', args.dataset_dir)
    print('Checkpoint directory:', args.ckpt_dir)
    print('Checkpoint interval:', args.ckpt_interval)
    print('Log directory:', args.log_dir)
    print('Pretrained model path:', args.model_path)
    print('Batch size:', args.batch_size)
    print('Timestep:', args.timestep)
    print('Epoch:', args.epoch)
    print('Learning rate:', args.learning_rate)
    print('Test overfit:', args.test_overfit)