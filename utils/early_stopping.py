import math
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, mode='loss', patience=7, verbose=False, delta=0, path='/checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth'
            trace_func (function): trace print function.
                            Default: print
            mode = 'loss'/'acc'
        """
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.min_score = None

    def __call__(self, model, score):

        if (self.mode == 'loss'):
            score = (- score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.min_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, value, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.mode == 'loss':
                self.trace_func(f'Validation loss decreased ({math.abs(self.min_score)} --> {math.abs(score)}).  Saving model ...')
            else:
                self.trace_func(f'Validation accuracy increased ({self.min_score} --> {score}).  Saving model ...')

        torch.save(model.state_dict(), self.path)
        self.min_score = value
