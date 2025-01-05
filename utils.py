class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        
    def __deepcopy__(self, memo):
        from copy import deepcopy
        return dotdict(deepcopy(dict(self), memo))
    
def get_latest_checkpoint(checkpoint_dir):
    """
    Finds the checkpoint with the highest iteration number in the given directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Path to latest checkpoint file, or None if no checkpoints found
    """
    import os
    import re

    # Get all checkpoint files
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth.tar')]
    
    if not files:
        return None
        
    # Extract iteration numbers using regex
    numbers = []
    for f in files:
        match = re.search(r'checkpoint_(\d+)\.pth\.tar', f)
        if match:
            numbers.append((int(match.group(1)), f))
            
    # Sort by iteration number and get the latest
    numbers.sort(reverse=True)
    return os.path.join(checkpoint_dir, numbers[0][1])
