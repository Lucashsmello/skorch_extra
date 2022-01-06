from sklearn.base import TransformerMixin
import skorch
import numpy as np
import torch
from skorch import NeuralNet
import skorch
import os
from tempfile import mkdtemp
from torch.nn import NLLLoss


class NeuralNetBase(NeuralNet):
    """
    Extends :class:`skorch.NeuralNet` mainly by including a functionality to cache/save neural network and load it later automatically.
    It uses hashing for checking if a model is already saved/cached.

    It also includes a way to manually set random_state. 
    """

    def __init__(self, module, *args, cache_dir=mkdtemp(), init_random_state=None, **kwargs):
        super().__init__(module, *args, **kwargs)
        self.init_random_state = init_random_state
        self.cache_dir = cache_dir

    def fit(self, X, y, **fit_params):
        if(isinstance(X, dict)):
            Xf = X['X']
        else:
            Xf = X
        if(self.cache_dir is not None):
            cache_filename = self.get_cache_filename(Xf, y)
            if(os.path.isfile(cache_filename)):
                if not self.warm_start or not self.initialized_:
                    self.initialize()
                print("loading cached neuralnet '%s'" % cache_filename)
                self.load_params(f_params=cache_filename)

                return self
            super().fit(X, y, **fit_params)

            # Only save if user did not interrupted the training process.
            if(len(self.history) == self.max_epochs):
                self.save_params(f_params=cache_filename)
            else:
                for _, cb in self.callbacks_:
                    if(isinstance(cb, skorch.callbacks.EarlyStopping)):
                        if(cb.misses_ == cb.patience):
                            self.save_params(f_params=cache_filename)
                            break

        else:
            super().fit(X, y, **fit_params)
        return self

    def initialize(self):
        if(self.init_random_state is not None):
            np.random.seed(self.init_random_state)
            torch.cuda.manual_seed(self.init_random_state)
            torch.manual_seed(self.init_random_state)
        return super().initialize()

    def get_cache_filename(self, X, y) -> str:
        import hashlib

        if(isinstance(X, dict)):
            Xf = X['X']
        else:
            Xf = X

        m = hashlib.md5()
        m.update(self.__class__.__name__.encode('utf-8'))
        n = len(Xf)

        for k, v in self.get_params().items():
            if(k == 'cache_dir'):
                continue
            if(type(v) in (int, float, bool, str)):
                s = '%s:%s' % (str(k), str(v))
            elif(type(v) in (dict, list, tuple)):
                s = '%s:%s' % (str(k), str(len(v)))
            elif(isinstance(v, type)):
                s = '%s:%s' % (str(k), v.__name__)
            else:
                s = '%s:%s' % (str(k), v.__class__.__name__)
            m.update(s.encode('utf-8'))
        m.update(str(n).encode('utf-8'))
        m.update(str(Xf[0]).encode('utf-8'))
        m.update(str(Xf[1]).encode('utf-8'))
        # m.update(str(Xf[:, 0]).encode('utf-8'))
        # m.update(str(Xf[0]).encode('utf-8'))
        # m.update(str(Xf[:len(Xf)//2, :].mean()).encode('utf-8'))
        # m.update(str(Xf[:len(Xf)//4, :10].mean()).encode('utf-8'))
        # m.update(str(Xf.max()).encode('utf-8'))
        # m.update(str(Xf.min()).encode('utf-8'))

        m.update(str(y[0]).encode('utf-8'))
        m.update(str(y.max()).encode('utf-8'))
        m.update(str(y[:n//2].sum()).encode('utf-8'))
        m.update(str(y[n//2:].sum()).encode('utf-8'))

        fname = m.hexdigest()
        return f"{self.cache_dir}/{fname}.pkl"

    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = skorch.utils.to_tensor(y_true, device=self.device)

        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_.train(training)

        loss = self.criterion_(y_pred, y_true)
        if(isinstance(loss, dict)):
            loss_dict = loss
            assert('loss' in loss_dict)
            for k, v in loss_dict.items():
                if(isinstance(v, dict)):
                    v = v['losses']
                if(k == 'loss'):
                    loss = v
                else:
                    name = "%s_%s" % ("train" if training else "valid", k)
                    self.history.record_batch(name, v.item())
        return loss


class NeuralNetTransformer(NeuralNetBase, TransformerMixin):
    def transform(self, X):
        return self.predict_proba(X)


class NeuralNetClassifier(NeuralNetBase, skorch.NeuralNetClassifier):
    def __init__(self, module, criterion=NLLLoss, *args, cache_dir=mkdtemp(), init_random_state=None, **kwargs):
        super().__init__(module, criterion=criterion, *args, cache_dir=cache_dir, init_random_state=init_random_state, **kwargs)

