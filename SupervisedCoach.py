import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Arena import Arena
from othello.OthelloPlayers import RandomPlayer
from NaiveSearch import NaiveSearch

log = logging.getLogger(__name__)


class SupervisedCoach():
    """
    This class executes the supervised learning from demonstrations. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        # self.mcts = MCTS(self.game, self.nnet, self.args)
        # self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        # self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def load_dataset(self, dataset_path):
        """
        Loads pre-collected training examples from a dataset file.
        
        Args:
            dataset_path: Path to the dataset file containing (board, policy, value) tuples
        
        Returns:
            List of training examples of list of (sequence of boards (T x board_size x board_size), sequence of actions (T))
        """
        log.info(f"Loading dataset from {dataset_path}")
        dataset = torch.load(dataset_path)

        return dataset

    def preprocess_dataset(self, dataset):
        """
        Preprocesses the dataset by removing the last board state from each trajectory.
        
        Args:
            dataset: List of (sequence of boards, sequence of actions) tuples
            
        Returns:
            Preprocessed dataset with last board state removed from each trajectory
        """
        all_boards = []
        all_actions = []

        for boards, actions in dataset:
            # Remove the last board state
            processed_boards = boards[:-1]

            # Divide the sequence of boards into T/2 subsequences for black and white (-1 and 1), black moves first
            black_boards = processed_boards[::2]
            white_boards = processed_boards[1::2]
            black_actions = actions[::2]
            white_actions = actions[1::2]

            # Get the canonical form of the board
            black_boards = self.game.getCanonicalForm(black_boards, -1)
            white_boards = self.game.getCanonicalForm(white_boards, 1) # unnecessary actually

            # Add the processed boards to the processed dataset
            all_boards.append(black_boards)
            all_boards.append(white_boards)
            all_actions.append(black_actions)
            all_actions.append(white_actions)

            # Get the symmetries of the board / augmentations
            if self.args.augment:

                for board, action in zip(black_boards, black_actions):
                    sym_black_boards_actions = self.game.getSymmetries(board, F.one_hot(action, num_classes=self.game.getActionSize())) # returns list of (board, action) tuples
                    sym_black_boards = [x[0] for x in sym_black_boards_actions]
                    sym_black_actions = [x[1] for x in sym_black_boards_actions]

                    all_boards.extend(sym_black_boards)
                    all_actions.extend(sym_black_actions)

                for board, action in zip(white_boards, white_actions):
                    sym_white_boards_actions = self.game.getSymmetries(board, F.one_hot(action, num_classes=self.game.getActionSize()))
                    sym_white_boards = [x[0] for x in sym_white_boards_actions]
                    sym_white_actions = [x[1] for x in sym_white_boards_actions]

                    all_boards.extend(sym_white_boards)
                    all_actions.extend(sym_white_actions)

        # return tensordataset
        board_tensor = torch.FloatTensor(np.concatenate(all_boards, axis=0))
        action_tensor = torch.LongTensor(np.concatenate(all_actions, axis=0))
        action_tensor = F.one_hot(action_tensor, num_classes=self.game.getActionSize())
        # Log the dataset size
        log.info(f"Dataset size: {len(board_tensor)}")
        return TensorDataset(board_tensor, action_tensor)


    def learn(self, start_iter=1):
        """
        Performs supervised learning on pre-collected dataset.
        """
        log.info('Starting supervised learning...')
        
        # Load the pre-collected dataset
        train_dataset = self.load_dataset(self.args.dataset_path)
        train_dataset = self.preprocess_dataset(train_dataset)

        # Add noise to the dataset
        if self.args.noise_ratio > 0.0:
            log.info(f"Adding noise to the dataset with ratio {self.args.noise_ratio}")
            noise_dataset = self.load_dataset(self.args.noise_dataset_path)
            noise_dataset = self.preprocess_dataset(noise_dataset)
            num_noise_examples = int(len(train_dataset) * self.args.noise_ratio)
        # randomly subset the tensor dataset if specified
        if self.args.subset_ratio < 1.0:
            indices = torch.randperm(len(train_dataset))[:int(len(train_dataset) * self.args.subset_ratio)]
            # Create subset of the dataset
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
            log.info(f"Using {int(self.args.subset_ratio * len(train_dataset))} examples ({self.args.subset_ratio * 100}% of original dataset)")

        # add noise to the dataset if specified
        if self.args.noise_ratio > 0.0:
            indices = torch.randperm(len(noise_dataset))[:num_noise_examples] # randomly subset the respective ratio number of examples from noise
            noise_dataset = torch.utils.data.Subset(noise_dataset, indices)
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, noise_dataset])
            log.info(f"Total dataset size: {len(train_dataset)} with {num_noise_examples} noise examples and {len(train_dataset) - num_noise_examples} original examples")
        # wrap in dataloader
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)

        # Training loop over epochs
        for epoch in range(self.args.epochs):
            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            print('EPOCH ::: ' + str(epoch + 1))
            epoch_loss = self.nnet.train(train_loader, epoch)
            log.info(f'Epoch {epoch + 1} average loss: {epoch_loss}')

            log.info('PITTING AGAINST PREVIOUS VERSION')

            nns = NaiveSearch(self.game, self.nnet, self.args)
            pns = NaiveSearch(self.game, self.pnet, self.args)


            arena = Arena(lambda x: np.argmax(pns.getActionProb(x)),
                          lambda x: np.argmax(nns.getActionProb(x)), self.game)
            pwins, nwins, draws, invalids = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d ; INVALID : %d' % (nwins, pwins, draws, invalids))

            # Add new arena comparison against random player
            log.info('PITTING AGAINST RANDOM PLAYER')
            rp = RandomPlayer(self.game).play
            random_arena = Arena(rp,
                               lambda x: np.argmax(nns.getActionProb(x)), self.game)
            rwins, nnwins, rdraws, rinvalids = random_arena.playGames(self.args.arenaCompare)
            log.info('NEW/RANDOM WINS : %d / %d ; DRAWS : %d ; INVALID : %d' % (nnwins, rwins, rdraws, rinvalids))

            # Log final metrics to wandb
            wandb.log({
                'epoch': epoch + 1,
                'arena_prev/new_model_wins': nwins,
                'arena_prev/draws': draws,
                'arena_prev/invalids': invalids,
                'arena_prev/win_rate': float(nwins) / (pwins + nwins) if (pwins + nwins) > 0 else 0,
                'arena_random/new_model_wins': nnwins,
                'arena_random/draws': rdraws,
                'arena_random/invalids': rinvalids,
                'arena_random/win_rate': float(nnwins) / (rwins + nnwins) if (rwins + nnwins) > 0 else 0,
                'epoch_loss': epoch_loss
            })
            # if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
            #     log.info('REJECTING NEW MODEL')
            #     self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # else:
            #     log.info('ACCEPTING NEW MODEL')
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(epoch))

            # Log final metrics to wandb
            wandb.log({
                'epoch': epoch + 1,
                'new_model_wins': nwins,
                'draws': draws,
                'win_rate_against_random': float(nwins) / (pwins + nwins) if (pwins + nwins) > 0 else 0,
                'epoch_loss': epoch_loss
            })

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            # r = input("Continue? [y|n]")
            # if r != "y":
            #     sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
