import logging
import numpy as np
import torch
import random
import argparse
import os

import coloredlogs
import wandb

from Coach import Coach
from SupervisedCoach import SupervisedCoach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from othello.pytorch.NNet import NNetWrapperSupervised as nn_sup
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# Expand argument parser
parser = argparse.ArgumentParser(description='Alpha Zero Training')
parser.add_argument('--supervised', action='store_true', default=False, help='Enable supervised learning mode')
parser.add_argument('--numIters', type=int, help='Number of iterations')
parser.add_argument('--numEps', type=int, help='Number of episodes per iteration')
parser.add_argument('--tempThreshold', type=int, help='Temperature threshold')
parser.add_argument('--updateThreshold', type=float, help='Update threshold for arena playoff')
parser.add_argument('--maxlenOfQueue', type=int, help='Max length of queue for training examples')
parser.add_argument('--numMCTSSims', type=int, help='Number of MCTS simulations')
parser.add_argument('--arenaCompare', type=int, help='Number of arena comparison games')
parser.add_argument('--cpuct', type=float, help='CPUCT value')
parser.add_argument('--augment', action='store_true', default=False, help='Enable data augmentation')
parser.add_argument('--checkpoint', type=str, help='Checkpoint directory')
parser.add_argument('--dataset_path', type=str, help='Path to dataset')
parser.add_argument('--subset_ratio', type=float, help='Ratio of training data to use (0.0-1.0)')
parser.add_argument('--project_name', type=str, help='wandb project name')
parser.add_argument('--experiment_name', type=str, help='wandb experiment name')
parser.add_argument('--seed', type=int, help='Random seed')
parser.add_argument('--batch_size', type=int, help='Training batch size')
parser.add_argument('--epochs', type=int, help='Number of training epochs')

# Parse command line arguments
cmd_args = parser.parse_args()

# Define default arguments
args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 20,
    'cpuct': 1,

    'augment': False,
    
    'checkpoint': './checkpoints/',
    'load_model': False,
    'dataset_path': './othello_trajectories.pt',
    'numItersForTrainExamplesHistory': 10,

    'project_name': 'alpha-zero-othello',
    'experiment_name': 'default',
    'seed': 42,

    'subset_ratio': 1.0,  # Use all data by default
    'batch_size': 512,
    'epochs': 10,
})

# Update args with any command-line specified values
for arg in vars(cmd_args):
    if getattr(cmd_args, arg) is not None:  # Only update if the argument was specified
        args[arg] = getattr(cmd_args, arg)

def main():
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    log.info('Loading %s...', Game.__name__)
    g = Game(8)

    log.info('Loading %s...', nn.__name__)
    if args.supervised:
        log.info('Supervised training enabled...')
        nnet = nn_sup(g)
    else:
        nnet = nn(g)
    
    # Add torch.compile for the neural network
    # if torch.cuda.is_available():
    #     log.info('Compiling neural network with torch.compile...')
    #     nnet.nnet = torch.compile(nnet.nnet)

    args.checkpoint = os.path.join(args.checkpoint, args.experiment_name)

    try:
        latest_checkpoint = get_latest_checkpoint(args.checkpoint)
        log.info('Loading checkpoint "%s"...', latest_checkpoint)

        # separate checkpoint path into folder and filename
        checkpoint_folder, checkpoint_filename = os.path.split(latest_checkpoint)
        start_iter = nnet.load_checkpoint(checkpoint_folder, checkpoint_filename)
        args.load_model = True
    except Exception as e:
        log.warning(f"Error loading checkpoint: {e}")
        log.warning("Continuing without loading a checkpoint.")
        start_iter = 1

    log.info('Loading the Coach...')
    if args.supervised:
        c = SupervisedCoach(g, nnet, args)
    else:
        c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples(latest_checkpoint)

    # Initialize wandb
    wandb.init(
        project=args.project_name,
        id=args.experiment_name,
        config=dict(args)
    )

    log.info('Starting the learning process 🎉')
    c.learn(start_iter=start_iter)


if __name__ == "__main__":
    main()
