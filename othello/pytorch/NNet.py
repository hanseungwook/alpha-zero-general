import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
import wandb

from .OthelloNNet import OthelloNNet as onnet
from .OthelloNNet import OthelloNNetSupervised as snet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True, # torch.cuda.is_available(),
    'num_channels': 512,
})

sup_args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    # 'epochs': 10,
    # 'batch_size': 128,
    'cuda': True,
    'num_channels': 512,
})

class NNetWrapperSupervised(NeuralNet):
    def __init__(self, game):
        self.nnet = snet(game, sup_args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.pi_losses = AverageMeter()

        if args.cuda:
            self.nnet.cuda()
        self.device = torch.device('cuda')
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=sup_args.lr)
    
    def train(self, train_loader, epoch):
        """
        train_loader: DataLoader containing (board, action) pairs
        """
        
        self.nnet.train()

        t = tqdm(train_loader, desc=f'Training Net Epoch {epoch + 1}')
        for step, (boards, target_actions) in enumerate(t):
            if args.cuda:
                boards, target_actions = boards.contiguous().to(self.device), target_actions.contiguous().to(self.device)

            # compute output
            out_pi = self.nnet(boards)
            loss = self.loss_pi(out_pi, target_actions)

            # record loss
            self.pi_losses.update(loss.item(), boards.size(0))
            t.set_postfix(Loss_pi=self.pi_losses)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate global step
            global_step = epoch * len(train_loader) + step

            # Log metrics to wandb
            wandb.log({
                'train/loss': loss.cpu().detach().item(),
                'train/step': global_step,
            })

        return self.pi_losses.avg

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        if not isinstance(board, torch.Tensor):
            board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda:
            board = board.contiguous().to(self.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.makedirs(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]
    


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # if args.cuda:
            # self.nnet.cuda()

        self.device = torch.device('cuda')

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        self.nnet.to(self.device)
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().to(self.device), target_pis.contiguous().to(self.device), target_vs.contiguous().to(self.device)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Log metrics to wandb
                wandb.log({
                    'train/policy_loss': l_pi.item(),
                    'train/value_loss': l_v.item(),
                    'train/total_loss': total_loss.item(),
                })

    def predict(self, board):
        """
        board: np array with board
        """

        self.nnet.to(self.device)

        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: 
            board = board.contiguous().to(self.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', iteration=None):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
        print("Checkpoint Directory does not exist! Making directory {}".format(folder))
        os.mkdir(folder)
    else:
        print("Checkpoint Directory exists!")
    
    checkpoint = {
        'state_dict': self.nnet.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise("No model in path {}".format(filepath))
    map_location = None if args.cuda else 'cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    self.nnet.load_state_dict(checkpoint['state_dict'])
    return checkpoint.get('iteration', None)  # Return iteration if exists, None otherwise