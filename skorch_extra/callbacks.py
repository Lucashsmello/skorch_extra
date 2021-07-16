from skorch import callbacks
import os
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime


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
            y = [self.labels_name[a] for a in y]

        # D_train, D_valid = net.get_split_datasets(X, y, **kwargs)

        X_emb = net.transform(X)
        self.writer.add_embedding(tag=self.foldtag+"/metric_space/train", mat=X_emb, metadata=y)

        if(net.validation_dataset is not None):  # FIXME: use net.get_split_datasets
            y_val = net.validation_dataset.y
            if(self.labels_name is not None):
                y_val = [self.labels_name[a] for a in y_val]
            X_emb = net.transform(net.validation_dataset.X)
            self.writer.add_embedding(tag=self.foldtag+"/metric_space/valid", mat=X_emb, metadata=y_val)
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
