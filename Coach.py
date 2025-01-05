import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import time

import numpy as np
from tqdm import tqdm, trange
import wandb
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import torch
from copy import deepcopy

from Arena import Arena
from MCTS import MCTS
from NaiveSearch import NaiveSearch
from othello.OthelloPlayers import RandomPlayer

log = logging.getLogger(__name__)


def selfplay_worker(game, nnet, args, queue, lock, counter, total_episodes, device_id):
    """
    Worker that runs self-play episodes on a specific GPU (device_id).

    :param game: Your Game instance.
    :param nnet: Your neural net instance.
    :param args: The same args used for MCTS, etc.
    :param queue: A multiprocessing.Queue for sending back results.
    :param lock: A multiprocessing.Lock to protect the shared counter.
    :param counter: A multiprocessing.Value tracking how many episodes have started so far.
    :param total_episodes: The total number of episodes to produce.
    :param device_id: The integer GPU ID this worker should use, e.g. 'cuda:0'.
    """
    # Add seed initialization at the start of the worker
    worker_seed = (int(time.time() * 1000) + device_id) % (2**32 - 1)  # ensure seed is within valid range
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
    
    device = torch.device(f'cuda:{device_id}')
    nnet.nnet.to(device)
    nnet.device = device

    while True:
        with lock:
            if counter.value >= total_episodes:
                return  # no more episodes to run
            episode_idx = counter.value
            counter.value += 1

        # Create a new MCTS using the network on this GPU
        mcts = MCTS(game, nnet, args)

        trainExamples = []
        board = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = game.getNextState(board, curPlayer, action)
            r = game.getGameEnded(board, curPlayer)

            if r != 0:
                final_examples = [
                    (x[0], x[2], r * ((-1)**(x[1] != curPlayer)))
                    for x in trainExamples
                ]
                queue.put(final_examples)
                break


def multiprocess_selfplay(game, nnet, args, num_episodes):
    """
    Spawns one process per GPU (or fewer, if num_episodes < num_gpus).
    Each process uses a distinct GPU to run self-play episodes until we have collected num_episodes.
    Returns a deque of all training examples.

    :param game: your Game instance.
    :param nnet: your neural net instance.
    :param args: same args as used by the MCTS.
    :param num_episodes: total episodes to run.
    :return: deque of training examples (board, pi, value).
    """
    num_gpus = torch.cuda.device_count()
    log.info(f"Number of GPUs: {num_gpus}")
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected for multi-GPU self-play.")

    # We'll create one process per GPU, or fewer if num_episodes < num_gpus
    num_processes = min(num_gpus, num_episodes)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    lock = manager.Lock()
    counter = manager.Value('i', 0)  # how many episodes have started

    processes = []
    log.info(f'Starting {num_processes} processes for self-play')
    for gpu_id in range(num_processes):
        p = multiprocessing.Process(
            target=selfplay_worker,
            args=(game, nnet, args, result_queue, lock, counter, num_episodes, gpu_id)
        )
        p.start()
        processes.append(p)

    # Collect results
    all_examples = []
    with tqdm(total=num_episodes, desc="Self Play (Multi-GPU)") as pbar:
        eps_collected = 0
        while eps_collected < num_episodes:
            examples = result_queue.get()  # blocking get
            all_examples.extend(examples)
            eps_collected += 1
            pbar.update(1)

    # Join processes
    for p in processes:
        p.join()

    return deque(all_examples, maxlen=args.maxlenOfQueue)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self, start_iter=1):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in trange(start_iter, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                # iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                # for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                #     self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                #     iterationTrainExamples += self.executeEpisode()

                # move nnet to cpu for multiprocessing
                self.nnet.nnet.cpu()
                # Collect examples via multiprocessing self-play
                iterationTrainExamples = multiprocess_selfplay(
                    self.game,
                    self.nnet,
                    self.args, 
                    self.args.numEps
                )
                # iterationTrainExamples = self.selfPlayManager.collect_examples(self.args.numEps)
                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='coach_temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='coach_temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws, invalids = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d ; INVALID : %d' % (nwins, pwins, draws, invalids))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='coach_temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            # Add new arena comparison against random player
            log.info('PITTING AGAINST RANDOM PLAYER')
            rp = RandomPlayer(self.game).play
            nns = NaiveSearch(self.game, self.nnet, self.args)
            random_arena = Arena(rp,
                               lambda x: np.argmax(nns.getActionProb(x)), self.game)
            rwins, nnwins, rdraws, rinvalids = random_arena.playGames(self.args.arenaCompare)
            log.info('NEW/RANDOM WINS : %d / %d ; DRAWS : %d ; INVALID : %d' % (nnwins, rwins, rdraws, rinvalids))

            # Log metrics to wandb
            wandb.log({
                'iteration': i,
                'arena_prev/new_model_wins': nwins,
                'arena_prev/draws': draws,
                'arena_prev/invalid': invalids,
                'arena_prev/win_rate': float(nwins) / (pwins + nwins) if (pwins + nwins) > 0 else 0,
                'arena_random/new_model_wins': nnwins,
                'arena_random/draws': rdraws,
                'arena_random/invalid': rinvalids,
                'arena_random/win_rate': float(nnwins) / (rwins + nnwins) if (rwins + nnwins) > 0 else 0
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

    def loadTrainExamples(self, modelFile):
        # modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
