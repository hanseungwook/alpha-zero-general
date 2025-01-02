import logging
import numpy as np
import torch
import random
import argparse

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
parser.add_argument('--checkpoint', type=str, help='Checkpoint directory')
parser.add_argument('--load_model', action='store_true', help='Whether to load a model')
parser.add_argument('--dataset_path', type=str, help='Path to dataset')
parser.add_argument('--project_name', type=str, help='wandb project name')
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
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'dataset_path': './othello_trajectories.pt',
    'numItersForTrainExamplesHistory': 20,

    'project_name': 'alpha-zero-othello',
    'seed': 42,

    'batch_size': 512,
    'epochs': 10,
})

# Update args with any command-line specified values
for arg in vars(cmd_args):
    if getattr(cmd_args, arg) is not None:  # Only update if the argument was specified
        args[arg] = getattr(cmd_args, arg)

# Add supervised flag from command line to args
args.supervised = cmd_args.supervised

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

    if args.load_model:
        try:
            log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        except Exception as e:
            log.warning(f"Error loading checkpoint: {e}")
            log.warning("Continuing without loading a checkpoint.")
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    if args.supervised:
        c = SupervisedCoach(g, nnet, args)
    else:
        c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    # Initialize wandb
    wandb.init(
        project=args.project_name,
        config=dict(args)
    )

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
