import numpy as np

class NaiveSearch():
    """
    This class implements a simple search that just uses the policy network's outputs
    directly, masking invalid moves.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

    def getActionProb(self, canonicalBoard):
        """
        Returns action probabilities based on the neural network's policy output.
        Invalid moves are masked and probabilities are renormalized.

        Args:
            canonicalBoard: The board in canonical form

        Returns:
            probs: a policy vector where the probability of invalid moves is 0
                  and valid moves are proportional to the network's output
        """
        # Get policy from neural network
        policy = self.nnet.predict(canonicalBoard)

        # if network returns policy and value, only use policy
        if isinstance(policy, tuple):
            policy = policy[0]
        
        # Get valid moves and mask invalid ones
        valids = self.game.getValidMoves(canonicalBoard, 1)
        policy = policy * valids

        # Renormalize probabilities
        sum_policy = np.sum(policy)
        if sum_policy > 0:
            policy /= sum_policy
        else:
            # If all valid moves were masked, make all valid moves equally probable
            policy = valids / np.sum(valids)

        return policy 