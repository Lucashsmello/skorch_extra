from typing import Tuple
import skorch
from skorch import callbacks
import os
from pathlib import Path
from skorch.net import NeuralNet
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import torch
import numpy as np


class LoadEndState(callbacks.Callback):
    """
    Loads weights from a checkpoint when training ends.
    This is useful, for example, to load and use the best weights of all epochs.
    """

    def __init__(self, checkpoint: callbacks.Checkpoint, delete_checkpoint=False):
        """
        Args:
            delele_checkpoints: Deletes checkpoint after loading it.
        """
        self.checkpoint = checkpoint
        self.delete_checkpoint = delete_checkpoint

    def on_train_end(self, net,
                     X=None, y=None, **kwargs):
        net.load_params(checkpoint=self.checkpoint)
        if(self.delete_checkpoint):
            os.remove(Path(self.checkpoint.dirname) / self.checkpoint.f_params)


class TensorBoardCallbackBase(callbacks.Callback):
    """
    Works with multiple folds.
    """
    UUIDs = {}
    SUMMARY_WRITERS = {}

    def __init__(self, writer, close_after_train=True):
        self.writer = writer
        self.close_after_train = close_after_train

    def on_train_begin(self, net, X, y, **kwargs):
        import hashlib
        if(isinstance(self.writer, dict)):
            w = str(self.writer)
            if(w in TensorBoardCallbackBase.SUMMARY_WRITERS):
                self.writer = TensorBoardCallbackBase.SUMMARY_WRITERS[w]
            else:
                self.writer = SummaryWriter(**self.writer)
                TensorBoardCallbackBase.SUMMARY_WRITERS[w] = self.writer
        m = hashlib.md5()
        m.update(str(X[:len(X)//2].mean()).encode('utf-8'))
        m.update(str(X.max()).encode('utf-8'))
        m.update(str(y[:len(y)//2].sum()).encode('utf-8'))
        uuid = m.hexdigest()
        if(uuid not in TensorBoardCallbackBase.UUIDs):
            TensorBoardCallbackBase.UUIDs[uuid] = len(TensorBoardCallbackBase.UUIDs)+1
        self.foldtag = "fold-%d" % TensorBoardCallbackBase.UUIDs[uuid]

        return super().on_train_begin(net, X=X, y=y, **kwargs)

    def get_params(self, deep):
        params = super().get_params(deep=deep)
        if(isinstance(self.writer, dict)):
            return params

        w = self.writer

        writer_params = {
            'log_dir': w.log_dir,
            'purge_step': w.purge_step,
            'max_queue': w.max_queue,
            'flush_secs': w.flush_secs,
            'filename_suffix': w.filename_suffix
        }
        params['writer'] = writer_params
        return params

    def on_train_end(self, net, X, y, **kwargs):
        if self.close_after_train:
            self.writer.close()

    @staticmethod
    def create_SummaryWriter(root_directory, name="") -> SummaryWriter:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(root_directory, current_time, name)
        return SummaryWriter(log_dir=log_dir)


class TensorBoardEmbeddingCallback(TensorBoardCallbackBase):
    """
    Callback that saves images of embeddings of a net.
    The neural net must implement transform(.) method.
    """

    def __init__(self, writer: SummaryWriter, close_after_train=False, labels_name=None) -> None:
        super().__init__(writer, close_after_train)
        self.labels_name = labels_name

    def on_train_end(self, net, X=None, y=None, **kwargs):
        if(self.labels_name is not None):
            y = [self.labels_name[int(a)] for a in y]

        # D_train, D_valid = net.get_split_datasets(X, y, **kwargs)

        X_emb = net.transform(X)
        self.writer.add_embedding(tag=self.foldtag+"/metric_space/train", mat=X_emb, metadata=y)

        # if(net.validation_dataset is not None):  # FIXME: use net.get_split_datasets
        #     y_val = net.validation_dataset.y
        #     if(self.labels_name is not None):
        #         y_val = [self.labels_name[a] for a in y_val]
        #     X_emb = net.transform(net.validation_dataset.X)
        #     self.writer.add_embedding(tag=self.foldtag+"/metric_space/valid", mat=X_emb, metadata=y_val)
        super().on_train_end(net, X=X, y=y, **kwargs)


class TensorBoardCallback(TensorBoardCallbackBase, callbacks.TensorBoard):
    def __init__(self, writer, close_after_train, keys_ignored=None, key_mapper=lambda x: x):
        callbacks.TensorBoard.__init__(self, writer, close_after_train=close_after_train,
                                       keys_ignored=keys_ignored, key_mapper=key_mapper)

    def add_scalar_maybe(self, history, key, tag, global_step):
        return super().add_scalar_maybe(history, key, self.foldtag+'/'+tag, global_step=global_step)


class ExtendedEpochScoring(callbacks.EpochScoring):
    """
    Enables EpochScoring to be run at training data and validation data simultaneously.
    """

    def get_test_data(self, dataset_train, dataset_valid):
        assert(self.use_caching == False), "Caching not available for ExtendedEpochScoring"
        on_train = self.on_train
        self.on_train = True
        Xtrain, ytrain, _ = super().get_test_data(dataset_train, dataset_valid)
        self.on_train = False
        Xvalid, yvalid, _ = super().get_test_data(dataset_train, dataset_valid)
        self.on_train = on_train
        return (Xtrain, Xvalid), (ytrain, yvalid), []

    def _record_score(self, history, current_score):
        # if(current_score is not tuple):
        #     super()._record_score(history, current_score)
        trainname = "train_"+self.name_
        validname = "valid_"+self.name_
        train_score, valid_score = current_score
        history.record(trainname, train_score)
        if(valid_score is not None):
            history.record(validname, valid_score)
            score = train_score if self.on_train else valid_score
        else:
            score = train_score

        is_best = self._is_best_score(score)
        if is_best is None:
            return
        # name = trainname if self.on_train else validname
        history.record(self.name_ + '_best', bool(is_best))
        if is_best:
            self.best_score_ = score


def _get_labels(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        labels = _get_labels(dataset.dataset)
        return labels[dataset.indices]

    """
    Guesses how to get the labels.
    """
    if hasattr(dataset, 'get_labels'):
        return dataset.get_labels()
    if hasattr(dataset, 'labels'):
        return dataset.labels
    if hasattr(dataset, 'targets'):
        return dataset.targets
    if hasattr(dataset, 'y'):
        return dataset.y

    import torchvision
    if isinstance(dataset, torchvision.datasets.MNIST):
        return dataset.train_labels.tolist()
    if isinstance(dataset, torchvision.datasets.ImageFolder):
        return [x[1] for x in dataset.imgs]
    if isinstance(dataset, torchvision.datasets.DatasetFolder):
        return dataset.samples[:][1]
    raise NotImplementedError("BalancedDataLoader: Labels were not found!")


class EstimatorEpochScoring(ExtendedEpochScoring):
    class EstimatorCallback:
        def __init__(self, estimator, metric, use_transform=True):
            from sklearn.metrics import get_scorer
            self.estimator = estimator
            self.metric = get_scorer(metric)
            self.use_transform = use_transform

        def __call__(self, net, X, y) -> Tuple[float, float]:
            Xtrain, Xvalid = X
            if(y[0] is None):
                ytrain = _get_labels(Xtrain)
                yvalid = _get_labels(Xvalid)
            else:
                ytrain, yvalid = y

            if(self.use_transform):
                X_emb = net.transform(Xtrain)
            else:
                X_emb = net.predict(Xtrain)
            self.estimator.fit(X_emb, ytrain)
            score_train = self.metric(self.estimator, X_emb, ytrain)

            if(Xvalid is not None):
                if(self.use_transform):
                    X_emb = net.transform(Xvalid)
                else:
                    X_emb = net.predict(Xvalid)
                score_valid = self.metric(self.estimator, X_emb, yvalid)
            else:
                score_valid = None
            return score_train, score_valid

    def __init__(self, estimator, metric='f1_macro', name='score', lower_is_better=False,
                 use_caching=False, on_train=False, use_transform=True,
                 **kwargs):
        self.estimator = estimator
        self.metric = metric
        est_cb = EstimatorEpochScoring.EstimatorCallback(estimator, metric, use_transform)
        super().__init__(est_cb, lower_is_better=lower_is_better, use_caching=use_caching, name=name, on_train=on_train,
                         **kwargs)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        del params['scoring']
        return params


class Ydiff:
    """
    Metric callback to be used with EpochScoring.
    Measures the difference in losses, for each respective batch or sample.
    """

    def __init__(self):
        self.last_pred = None

    def __call__(self, net, X, y=None):
        """
        When caching enabled, y seems to be shuffled, but X is not? It seems to be a bug.
        """
        yp = net.predict_proba(X)
        if(self.last_pred is not None):
            Dists = ((self.last_pred - yp)**2).mean(axis=1)
            ret = (Dists**0.5).mean()
        else:
            ret = np.nan
        self.last_pred = yp

        return ret


def calculate_loss(criterion, dataset):
    with torch.no_grad():
        loss_list = []
        for ypi, yi in dataset:
            l = criterion(torch.FloatTensor(ypi), torch.LongTensor(yi)).numpy()
            if(hasattr(l, 'shape') and len(l.shape) > 0):
                loss_list.extend(l)
            else:
                loss_list.append(l)
    return np.array(loss_list)


class LossDiff:
    """
    Metric callback to be used with EpochScoring.
    Measures the difference in losses, for each respective batch or sample.
    Please, dont use caching=True with this loss callback
    """

    def __init__(self, criterion=None, random_state=None,
                 diff_calback=lambda L1, L2: np.nan if len(L1) == 0 else np.linalg.norm(L1-L2)/len(L1),
                 filter_callback=None, training=True) -> None:
        self.last_losses = None
        self.random_state = random_state
        self.criterion = criterion
        self.diff_calback = diff_calback
        self.filter_callback = filter_callback
        self.training = training

    def __call__(self, net: NeuralNet, X, y=None):
        if(self.random_state is not None):
            torch.random.manual_seed(self.random_state)
            torch.cuda.random.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        yp = torch.FloatTensor(net.predict_proba(X))
        if(isinstance(X, torch.utils.data.Dataset)):
            y = torch.LongTensor([X[i][1] for i in range(len(X))])
        else:
            y = torch.LongTensor(y)

        criterion = net.criterion_ if self.criterion is None else self.criterion
        it = net.get_iterator(skorch.dataset.Dataset(yp, y), training=self.training)
        loss_list = calculate_loss(criterion, it)
        if(self.last_losses is not None):
            if(self.filter_callback is not None):
                mask = self.filter_callback(self.last_losses, loss_list)
                L1, L2 = self.last_losses[mask], loss_list[mask]
            else:
                L1, L2 = self.last_losses, loss_list
            ret = self.diff_calback(L1, L2)
        else:
            ret = np.nan
        self.last_losses = loss_list

        return ret


class HardLosses:
    def __init__(self, hard_threshold='p95', training=True) -> None:
        self.hard_thresdhold = hard_threshold
        self.training = training

    def __call__(self, net: skorch.NeuralNet, X, y=None):
        yp = torch.FloatTensor(net.predict_proba(X))
        if(isinstance(X, torch.utils.data.Dataset)):
            y = torch.LongTensor([X[i][1] for i in range(len(X))])
        else:
            y = torch.LongTensor(y)

        criterion = net.criterion_
        it = net.get_iterator(skorch.dataset.Dataset(yp, y), training=self.training)
        loss_list = calculate_loss(criterion, it)
        if(isinstance(self.hard_thresdhold, str)):
            p = np.percentile(loss_list, int(self.hard_thresdhold[1:3]))
            loss_list = loss_list[loss_list >= p]
        else:
            loss_list = loss_list[loss_list >= self.hard_thresdhold]

        return loss_list.mean()

