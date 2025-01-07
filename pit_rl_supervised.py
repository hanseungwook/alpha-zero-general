import Arena
from MCTS import MCTS
from NaiveSearch import NaiveSearch
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
from othello.pytorch.NNet import NNetWrapperSupervised as NNetSupervised


import numpy as np
from utils import *
import logging
import coloredlogs

# Set up logger for this module
log = logging.getLogger(__name__)

# Optional: Configure coloredlogs for prettier logging output
coloredlogs.install(level='INFO')  # You can change to DEBUG for more verbose output

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.

if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)


# nnet players
n1 = NNet(g)
n2 = NNetSupervised(g)

if mini_othello:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
    n2.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_supervised_best.pth.tar')
else:
    # n1.load_checkpoint('./temp/','best.pth.tar')
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
    
    # n2.load_checkpoint('./checkpoints/sft_rerun2/','temp.pth.tar')
    # n2.load_checkpoint('./checkpoints/sft_subset0.75/', 'checkpoint_9.pth.tar')
    n2.load_checkpoint('./checkpoints/sft_noise_0.4/', 'checkpoint_9.pth.tar')

# Setup MCTS for both networks
args = dotdict({'numMCTSSims': 100, 'cpuct':1.0})

mcts = MCTS(g, n1, args)
ns = NaiveSearch(g, n2, args)

n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
n2p = lambda x: np.argmax(ns.getActionProb(x))

arena = Arena.Arena(n1p, n2p, g)

wins1, wins2, draws, invalids = arena.playGames(100)

print(f"Wins for RL: {wins1}, Wins for Supervised: {wins2}, Draws: {draws}, Invalids: {invalids}")