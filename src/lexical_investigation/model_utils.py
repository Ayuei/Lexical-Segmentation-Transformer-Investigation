import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_transformers import *
import torch.nn.init as init
from torch.nn import CrossEntropyLoss
import os
import logging
import math
from utils.VDCNN import ResidualBlock, KMaxPool
from RAdam.radam import RAdam
from pytorch_lamb.lamb import Lamb
from fastai.text.transform import *
from fastai.text.data import *
from fastai.core import *
from fastai.basic_data import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
from fastai.metrics import *
from pytorch_transformers import BertTokenizer, BertConfig, BertModel
import torch
import torch.nn as nn
from typing import *
from fastai.core import BatchSamples

def bert_pad_collate(samples: BatchSamples, pad_idx: int = 1, pad_first: bool = True, backwards: bool = False,
                     max_len: int = 64) -> Tuple[LongTensor, LongTensor]:
    "Function that collect samples and adds padding. Flips token order if needed"
    samples = to_data(samples)
    # max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    if backwards: pad_first = not pad_first
    for i, s in enumerate(samples):
        if pad_first:
            res[i, -len(s[0]):] = LongTensor(s[0])
        else:
            res[i, :len(s[0]):] = LongTensor(s[0])
    if backwards: res = res.flip(1)
    return res, tensor(np.array([s[1] for s in samples]))


class fbeta_binary(Callback):
    "Computes the f_beta between preds and targets for binary text classification"

    def __init__(self, beta2=1, eps=1e-9, sigmoid=True):
        self.beta2 = beta2 ** 2
        self.eps = eps
        self.sigmoid = sigmoid

    def on_epoch_begin(self, **kwargs):
        self.TP = 0
        self.total_y_pred = 0
        self.total_y_true = 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        y_pred = last_output
        y_pred = y_pred.softmax(dim=1)
        y_pred = y_pred.argmax(dim=1)
        y_true = last_target.float()

        self.TP += ((y_pred == 1) * (y_true == 1)).float().sum()
        self.total_y_pred += (y_pred == 1).float().sum()
        self.total_y_true += (y_true == 1).float().sum()

    def on_epoch_end(self, **kwargs):
        prec = self.TP / (self.total_y_pred + self.eps)
        rec = self.TP / (self.total_y_true + self.eps)
        res = (prec * rec) / (prec * self.beta2 + rec + self.eps) * (1 + self.beta2)
        # self.metric = res.mean()
        self.metric = res


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class BertTokenizerWrapper(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""

    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int = 128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t: str) -> List[str]:
        """Limits the maximum sequence length"""
        # return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]
        # Override the tokeniser aspect

        tokens = t.split()

        if len(tokens) < self.max_seq_len:
            while len(tokens) < self.max_seq_len:
                tokens.append('[PAD]')  # Pad at the end

        assert len(tokens) == self.max_seq_len
        return tokens


class BertTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)


class BertNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=Vocab(list(bert_tok.vocab.keys())), **kwargs)


def get_bert_processor(tokenizer: Tokenizer = None, vocab: Vocab = None):
    """
    Constructing preprocessors for BERT
    We remove sos/eos tokens since we add that ourselves in the tokenizer.
    We also use a custom vocabulary to match the numericalization with the original BERT model.
    """
    return [BertTokenizeProcessor(tokenizer=tokenizer),
            NumericalizeProcessor(vocab=vocab)]


class BertDataBunch(TextDataBunch):
    @classmethod
    def from_df(cls, path: PathOrStr, train_df: DataFrame, valid_df: DataFrame, test_df: Optional[DataFrame] = None,
                tokenizer: Tokenizer = None, vocab: Vocab = None,
                classes: Collection[str] = None, text_cols: IntsOrStrs = 1,
                label_cols: IntsOrStrs = 0, label_delim: str = None, **kwargs) -> DataBunch:
        p_kwargs, kwargs = split_kwargs_by_func(kwargs, get_bert_processor)
        # use our custom processors while taking tokenizer and vocab as kwargs
        processor = get_bert_processor(tokenizer=tokenizer, vocab=vocab, **p_kwargs)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_for_lm() if cls == TextLMDataBunch else src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)


def get_preds_as_nparray(ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in databunch.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]


class BertLearner(Learner):

    # https://github.com/huggingface/pytorch-pretrained-BERT/issues/95
    def unfreeze_all_layers(self) -> None:
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def freeze_embeddings(self) -> None:
        for name, param in self.model.named_parameters():
            # FIXME: check if any batchnorm layer present, set to False
            if ('embeddings' in name) or ('LayerNorm' in name):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def freeze_encoders_to(self, n=12) -> None:
        for name, param in self.model.named_parameters():
            index = 100000
            if 'encoder' in name:
                index = [int(s) for s in name.split(".") if s.isdigit()][0]

            if ('embeddings' in name) or ('LayerNorm' in name) or index < n:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def freeze_all_layers(self):
        for name, param in self.model.bert.named_parameters():
            param.requires_grad = False

    def print_trainable_layers(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad: print(name)

    def get_ordered_preds(self, ds_type: DatasetType = DatasetType.Valid, with_loss: bool = False,
                          n_batch: Optional[int] = None, pbar: Optional[PBar] = None,
                          ordered: bool = True) -> List[Tensor]:
        "Return predictions and targets on the valid, train, or test set, depending on `ds_type`."
        # FIXME: check if this is required. reset is done in fastai. implement if require for BERT also
        # learner.model.reset()
        self.model.eval()
        if ordered: np.random.seed(42)
        preds = self.get_preds(ds_type=ds_type, with_loss=with_loss, n_batch=n_batch, pbar=pbar)
        if ordered and hasattr(self.dl(ds_type), 'sampler'):
            np.random.seed(42)
            sampler = [i for i in self.dl(ds_type).sampler]
            reverse_sampler = np.argsort(sampler)
            preds = [p[reverse_sampler] for p in preds]
        return (preds)

    def get_predictions(self, ds_type: DatasetType = DatasetType.Valid, with_loss: bool = False,
                        n_batch: Optional[int] = None, pbar: Optional[PBar] = None,
                        ordered: bool = True):
        preds, true_labels = self.get_ordered_preds(ds_type=ds_type, with_loss=with_loss, n_batch=n_batch, pbar=pbar,
                                                    ordered=ordered)
        pred_values = np.argmax(preds, axis=1)
        return preds, pred_values, true_labels

    def print_metrics(self, preds, pred_values, true_labels):
        acc = accuracy(preds, true_labels)
        f1s = f1_score(true_labels, pred_values)
        print(f"Accuracy={acc}, f1_score={f1s}")

    def load_best_model(self, model_name="bestmodel"):
        try:
            self.load(model_name, purge=False)
            print(f"Loading {model_name}")
        except:
            print(f"Failed to load {model_name}")

    def similar(self, text):
        cls, _, _ = self.predict(text)
        return cls.obj == 1


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, first_stride=1, act_func=nn.ReLU):
        super(ConvolutionalBlock, self).__init__()

        padding = int((kernel_size - 1) / 2)
        if kernel_size == 3: assert padding == 1
        if kernel_size == 5: assert padding == 2
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=first_stride, padding=padding),
            nn.BatchNorm1d(num_features=out_channels)
        ]

        if act_func is not None:
            layers.append(act_func())

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class KMaxPool(nn.Module):
    def __init__(self, k='half'):
        super(KMaxPool, self).__init__()

        self.k = k

    def forward(self, x):
        # x : batch_size, channel, time_steps
        if self.k == 'half':
            time_steps = x.shape(2)
            self.k = time_steps // 2

        kmax, kargmax = torch.topk(x, self.k)
        # kmax, kargmax = x.topk(self.k, dim=2)
        return kmax


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, optional_shortcut=True,
                 kernel_size=1, act_func=nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.optional_shortcut = optional_shortcut
        self.convolutional_block = ConvolutionalBlock(in_channels, out_channels, first_stride=1,
                                                      act_func=act_func, kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        x = self.convolutional_block(x)

        if self.optional_shortcut:
            x = x + residual

        return x


class CNN_Bert(nn.Module):
    def __init__(self, bert_model, device: str, hidden_dim: int, num_layers: int, seq_length: int,
                 num_labels: int, k: int = 1, optional_shortcut: bool = True, hidden_neurons: int = 2048,
                 use_batch_norms: bool = True, use_trans_blocks: bool = False, residual_kernel_size: int = 1,
                 dropout_perc: float = 0.5, act_func=nn.ReLU):

        super().__init__()
        self.bert = bert_model
        self.device = device
        self.bert.to(self.device)
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.use_trans_blocks = use_trans_blocks
        self.use_batch_norms = use_batch_norms
        self.num_layers = num_layers

        # CNN Part

        conv_layers = []
        transformation_blocks = [None]  # Pad the first element, for the for loop in forward
        batchnorms = [None]  # Pad the first element

        # Adds up to num_layers + 1 embedding layer
        conv_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1))

        for i in range(2, 14):
            # Instead of downsampling, do a concat, and then try summing later
            # Try compressing the hidden dim with a 1x1 conv
            conv_layers.append(ResidualBlock(hidden_dim, hidden_dim, optional_shortcut=optional_shortcut,
                                             kernel_size=residual_kernel_size, act_func=act_func))
            if use_trans_blocks: transformation_blocks.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1))
            if use_batch_norms: batchnorms.append(nn.BatchNorm1d(hidden_dim))

        self.conv_layers = nn.ModuleList(conv_layers)
        if use_trans_blocks: self.transformation_blocks = nn.ModuleList(transformation_blocks)
        if use_batch_norms: self.batchnorms = nn.ModuleList(batchnorms)
        self.kmax_pooling = KMaxPool(k)

        linear_layers = []
        linear_layers.append(nn.Linear((hidden_dim) * k, hidden_neurons))  # Downsample into Kmaxpool?
        linear_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
        linear_layers.append(nn.Dropout(dropout_perc))
        linear_layers.append(nn.Linear(hidden_neurons, num_labels))

        self.linear_layers = nn.Sequential(*linear_layers)
        self.apply(weight_init)
        self.num_labels = num_labels

    def init_weights(self):
        self.apply(weight_init)
        self.bert.apply(self.bert.init_weights)

    def forward(self, *args, **kwargs):
        # 1xseq_lengthx768

        labels = kwargs['labels'] if 'labels' in kwargs else None
        if labels is not None: del kwargs['labels']

        bert_outputs = self.bert(*args, **kwargs)
        hidden_states = bert_outputs[-1]

        # Fix this, also draw out what ur model should do first
        is_embedding_layer = True

        assert len(self.conv_layers) == len(
            hidden_states)  # == len(self.transformation_blocks) == len(self.batchnorms), info

        zip_args = [self.conv_layers, hidden_states]
        identity_func = lambda k: k

        if self.use_trans_blocks:
            assert len(self.transformation_blocks) == len(hidden_states)
            zip_args.append(self.transformation_blocks)
        else:
            zip_args.append([identity_func for i in range(self.num_layers)])

        if self.use_batch_norms:
            assert len(self.batchnorms) == len(hidden_states)
            zip_args.append(self.batchnorms)
        else:
            zip_args.append([identity_func for i in range(self.num_layers)])

        for co, hi, tr, bn in zip(*zip_args):
            if is_embedding_layer:
                out = co(hi.transpose(1, 2))  # batchxhiddenxseq
                is_embedding_layer = not is_embedding_layer
            else:
                out = co(out + tr(bn(hi.transpose(1, 2))))  # add hidden dims

        assert out.shape[2] == self.seq_length

        out = self.kmax_pooling(out)
        logits = self.linear_layers(out.view(out.size(0), -1))
        outputs = logits

        return outputs
