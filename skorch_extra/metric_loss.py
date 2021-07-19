from pytorch_metric_learning.miners import BaseMiner
import torch


class MetricLearningLossWrapper(torch.nn.Module):
    """
    Wrapper for skorch to use desired metric loss function and miner.
    """

    def __init__(self, miner: BaseMiner, loss_func=None, loss_func_cls=None, loss_func_params=None):
        super().__init__()
        if(loss_func is None):
            self.loss_func = loss_func_cls(**loss_func_params)
        else:
            self.loss_func = loss_func
        self.miner = miner

    def forward(self, embeddings, target):
        if(self.miner is None):
            loss = self.loss_func(embeddings, target)
        else:
            idxs = self.miner(embeddings, target)
            loss = self.loss_func(embeddings, target, idxs)
        return loss

    def __call__(self, embeddings, target):
        return self.forward(embeddings, target)
