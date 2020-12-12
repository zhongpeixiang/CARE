"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax
from onmt.modules.util_class import Cast


def build_loss_compute(model, tgt_field, opt, train=True):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the NMTLossCompute class handles all loss computation except
    for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]

    if opt.lambda_coverage != 0:
        assert opt.coverage_attn, "--coverage_attn needs to be set in " \
            "order to use --lambda_coverage != 0"

    if opt.copy_attn:
        criterion = onmt.modules.CopyGeneratorLoss(
            len(tgt_field.vocab), opt.copy_attn_force,
            unk_index=unk_idx, ignore_index=padding_idx
        )
    elif opt.label_smoothing > 0 and train:
        if opt.decoder_type == "kg_transformer":
            criterion = KGLabelSmoothingLoss(
                opt.label_smoothing, len(tgt_field.vocab), ignore_index=padding_idx, num_steps=opt.train_steps
            )
        else:
            criterion = LabelSmoothingLoss(
                opt.label_smoothing, len(tgt_field.vocab), ignore_index=padding_idx
            )
    elif isinstance(model.generator[-1], LogSparsemax):
        criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
    else:
        criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    # if the loss function operates on vectors of raw logits instead of
    # probabilities, only the first part of the generator needs to be
    # passed to the NMTLossCompute. At the moment, the only supported
    # loss function of this kind is the sparsemax loss.
    use_raw_logits = isinstance(criterion, SparsemaxLoss)
    if opt.decoder_type != "eds":
        loss_gen = model.generator[0] if use_raw_logits else model.generator
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            criterion, loss_gen, tgt_field.vocab, opt.copy_loss_by_seqlength,
            lambda_coverage=opt.lambda_coverage
        )
    
    if opt.decoder_type == "eds":
        compute = EDSLossCompute(
            criterion, model.generator, lambda_coverage=opt.lambda_coverage, decoder=model.decoder, lambda_=opt.lambda_)
    elif opt.decoder_type == "kg_transformer":
        compute = KGLossCompute(
            criterion, loss_gen, lambda_coverage=opt.lambda_coverage, coefficient=opt.boc_coefficient, step_size=opt.boc_step_size)
    else:
        compute = NMTLossCompute(
            criterion, loss_gen, lambda_coverage=opt.lambda_coverage)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, batch_idx, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def __call__(self,
                 batch,
                 output,
                 attns,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None,
                 batch_idx=0):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        if trunc_size is None:
            trunc_size = batch.tgt.size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, output, trunc_range, attns)
        if shard_size == 0:
            # print("shard_size = 0")
            loss, stats = self._compute_loss(batch, batch_idx, **shard_state)
            return loss / float(normalization), stats
        batch_stats = onmt.utils.Statistics()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, batch_idx, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator, normalization="sents",
                 lambda_coverage=0.0):
        super(NMTLossCompute, self).__init__(criterion, generator)
        self.lambda_coverage = lambda_coverage

    def _make_shard_state(self, batch, output, range_, attns=None):
        shard_state = {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
        }
        if self.lambda_coverage != 0.0:
            coverage = attns.get("coverage", None)
            std = attns.get("std", None)
            assert attns is not None
            assert std is not None, "lambda_coverage != 0.0 requires " \
                "attention mechanism"
            assert coverage is not None, "lambda_coverage != 0.0 requires " \
                "coverage attention"

            shard_state.update({
                "std_attn": attns.get("std"),
                "coverage_attn": coverage
            })
        return shard_state

    def _compute_loss(self, batch, batch_idx, output, target, std_attn=None,
                      coverage_attn=None):
        # output: [tgt_len x batch x hidden]
        bottled_output = self._bottle(output) 

        scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)
        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(
                std_attn=std_attn, coverage_attn=coverage_attn)
            loss += coverage_loss
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats

    def _compute_coverage_loss(self, std_attn, coverage_attn):
        covloss = torch.min(std_attn, coverage_attn).sum()
        covloss *= self.lambda_coverage
        return covloss

class EDSLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator, normalization="sents",
                 lambda_coverage=0.0, decoder=None, lambda_=0):
        super(EDSLossCompute, self).__init__(criterion, generator)
        self.lambda_coverage = lambda_coverage
        self.decoder = decoder
        self.criterion_classification = nn.CrossEntropyLoss(reduction='sum')
        self.lambda_ = lambda_

    def _make_shard_state(self, batch, output, range_, attns=None):
        shard_state = {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
        }
        if self.lambda_coverage != 0.0:
            coverage = attns.get("coverage", None)
            std = attns.get("std", None)
            assert attns is not None
            assert std is not None, "lambda_coverage != 0.0 requires " \
                "attention mechanism"
            assert coverage is not None, "lambda_coverage != 0.0 requires " \
                "coverage attention"

            shard_state.update({
                "std_attn": attns.get("std"),
                "coverage_attn": coverage
            })
        return shard_state

    def get_emotion_mask(self, batch_emotion):
        batch_size = len(batch_emotion)
        emotion_mask = torch.zeros((batch_size, self.decoder.vocab_size)).to(batch_emotion.device)
        for i in range(batch_size):
            emotion_mask[i, self.decoder.emotion_vocab_indices[batch_emotion[i]]] = 1
        return emotion_mask
  

    def _compute_loss(self, batch, batch_idx, output, target, std_attn=None,
                      coverage_attn=None):
        tgt_len, batch_size, hidden_size = output.size()

        ###########
        ###########
        # method 0: vanilla probs
        if self.decoder.eds_type in [0]:
            scores = self.generator[0](self._bottle(output))
            # scores = self.generator(bottled_output) # vanilla loss

        ###########
        ###########
        # method 1: mask on logits, other emotional words have zero logits
        elif self.decoder.eds_type in [1]:
            logits = F.relu(self.generator[0][:2](output)) # [tgt_len, batch, vocab], relu refers to the official code
            type_controller = torch.sigmoid(self.generator[1](output)) # [tgt_len, batch, 1]
            mask = self.decoder.generic_mask.unsqueeze(0).unsqueeze(0) * type_controller + \
                self.get_emotion_mask(batch.emotion).unsqueeze(0) * (1-type_controller)
            logits = mask * logits
            scores = self.generator[0][2](self._bottle(logits)) # [tgt_len * batch, vocab]
        
        ###########
        ###########
        # method 2: two separate generators for generic and emotional words
        # generic vocab = vocab_size - emotion_vocab_size
        elif self.decoder.eds_type in [2]:
            all_probs = torch.zeros((tgt_len, batch_size, self.decoder.vocab_size)).type_as(output)
            generic_probs = torch.softmax(self.generator[0][:2](output), dim=-1) # [tgt_len, batch, generic_vocab]
            emotion_probs = torch.softmax(self.generator[1][:2](output), dim=-1) # [tgt_len, batch, emotion_vocab]
            type_controller = torch.sigmoid(self.generator[2](output)) # [tgt_len, batch, 1]
            generic_probs = (1-type_controller)*generic_probs
            emotion_probs = type_controller*emotion_probs
            for i in range(batch_size):
                generic_indices = torch.cat([self.decoder.generic_vocab_indices, self.decoder.other_emotion_indices[batch.emotion[i]]])
                all_probs[:,i,generic_indices] = generic_probs[:,i,:] # [tgt_len, generic_vocab]
                all_probs[:,i,self.decoder.emotion_vocab_indices[batch.emotion[i]]] = emotion_probs[:,i,:] # [tgt_len, emotion_vocab]
            all_probs = all_probs.view(tgt_len*batch_size, -1)
            scores = torch.log(all_probs)

        ###########
        ###########
        # method 3: two separate generators for generic and emotional words
        # emotion vocab = vocab_size - generic_vocab
        elif self.decoder.eds_type in [3]:
            all_probs = torch.zeros((tgt_len, batch_size, self.decoder.vocab_size)).type_as(output)
            generic_probs = torch.softmax(self.generator[0][:2](output), dim=-1) # [tgt_len, batch, generic_vocab]
            emotion_probs = torch.softmax(self.generator[1][:2](output), dim=-1) # [tgt_len, batch, emotion_vocab]
            type_controller = torch.sigmoid(self.generator[2](output)) # [tgt_len, batch, 1]
            generic_probs = (1-type_controller)*generic_probs
            emotion_probs = type_controller*emotion_probs
            all_probs[:,:,self.decoder.generic_vocab_indices] = generic_probs # [tgt_len, batch, generic_vocab]
            all_probs[:,:,self.decoder.all_emotion_indices] = emotion_probs # [tgt_len, batch, emotion_vocab]
            all_probs = all_probs.view(tgt_len*batch_size, -1)
            scores = torch.log(all_probs)
        
        # NLL Loss for Conversation Modelling
        gtruth = target.view(-1)
        loss = self.criterion(scores, gtruth)
        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(
                std_attn=std_attn, coverage_attn=coverage_attn)
            loss += coverage_loss
        stats = self._stats(loss.clone(), scores, gtruth)


        # Emotion Classification
        if not self.decoder.no_clf_loss:
            ###########
            ###########
            # method 0
            if self.decoder.eds_type in [0]:
                probs = torch.exp(scores) # [tgt_len * batch, vocab]
                total_embedding = torch.mm(probs, self.decoder.vocab_embedding).reshape(tgt_len, batch_size, -1).mean(dim=0) # [batch, emb_size]

            ###########
            ###########
            # method 1
            elif self.decoder.eds_type in [1]:
                probs = torch.exp(scores) # [tgt_len * batch, vocab]
                total_embedding = torch.mm(probs, self.decoder.vocab_embedding).reshape(tgt_len, batch_size, -1).mean(dim=0) # [batch, emb_size]

            ###########
            ###########
            # # method 2 and method 3
            elif self.decoder.eds_type in [2, 3]:
                total_embedding = torch.mm(all_probs, self.decoder.vocab_embedding).reshape(tgt_len, batch_size, -1).mean(dim=0) # [batch, emb_size]

            emotion_class_logits = self.decoder.emotion_classifier(total_embedding) # (batch, num_emotions)
            clf_loss = self.criterion_classification(emotion_class_logits, batch.emotion) # range: 1-2

            if random.random() < 0.005:
                print(loss, clf_loss)
            
            loss = loss + self.lambda_ * clf_loss

        return loss, stats


    def _compute_coverage_loss(self, std_attn, coverage_attn):
        covloss = torch.min(std_attn, coverage_attn).sum()
        covloss *= self.lambda_coverage
        return covloss


class KGLabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100, num_steps=1e5):
        super(KGLabelSmoothingLoss, self).__init__()
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.tgt_vocab_size = tgt_vocab_size
        self.top_k = 100
        self.confidence = 1.0 - label_smoothing
        topk_smoothing_percentage = 0.05 # 0.05, 0.8
        
        # smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        # smoothing_value = (1-topk_smoothing_percentage)*label_smoothing / (tgt_vocab_size - 2 - self.top_k)
        # one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        # self.register_buffer('one_hot', one_hot.unsqueeze(0))
        
        # self.confidence_topk = topk_smoothing_percentage*label_smoothing / self.top_k

        self.start_smoothing_value = label_smoothing/(tgt_vocab_size - 2)
        self.end_smoothing_value = (1-topk_smoothing_percentage)*label_smoothing/(tgt_vocab_size - 2 - self.top_k)
        self.step_size = (self.end_smoothing_value-self.start_smoothing_value)/num_steps
        self.topk_start_smoothing_value = label_smoothing/(tgt_vocab_size - 2)
        self.topk_end_smoothing_value = topk_smoothing_percentage*label_smoothing/self.top_k
        self.topk_step_size = (self.topk_end_smoothing_value-self.topk_start_smoothing_value)/num_steps

    def forward(self, output, target, concepts, batch_idx):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        concepts (LongTensor): (batch_size, top_k)
        batch_idx (int): the training step
        """
        # if batch_idx%1000 == 0:
        #     print((self.start_smoothing_value + batch_idx*self.step_size)*(self.tgt_vocab_size-2-self.top_k))
        #     print((self.topk_start_smoothing_value + batch_idx*self.topk_step_size)*self.top_k)
        
        # dynamic LS
        one_hot = torch.full((self.tgt_vocab_size,), self.start_smoothing_value + batch_idx*self.step_size).to(output.device)
        one_hot[self.ignore_index] = 0
        model_prob = one_hot.repeat(target.size(0), 1) # model_prob: (batch_size, vocab)
        model_prob.scatter_(1, concepts, self.topk_start_smoothing_value + batch_idx*self.topk_step_size)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence) 
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        # fixed LS
        # model_prob = self.one_hot.repeat(target.size(0), 1) # model_prob: (batch_size, vocab)
        # # print(model_prob.shape, output.shape, target.shape, concepts.shape)
        # model_prob.scatter_(1, concepts, self.confidence_topk)
        # model_prob.scatter_(1, target.unsqueeze(1), self.confidence) 
        # model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        # original
        # model_prob = self.one_hot.repeat(target.size(0), 1)
        # model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        # model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class KGLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator, normalization="sents",
                 lambda_coverage=0.0, coefficient=0, step_size=0):
        super(KGLossCompute, self).__init__(criterion, generator)
        self.lambda_coverage = lambda_coverage
        self.coefficient = coefficient
        self.step_size = step_size
        self.boc_criterion = nn.MSELoss(reduction='sum')

    def _make_shard_state(self, batch, output, range_, attns=None):
        shard_state = {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
        }
        if self.lambda_coverage != 0.0:
            coverage = attns.get("coverage", None)
            std = attns.get("std", None)
            assert attns is not None
            assert std is not None, "lambda_coverage != 0.0 requires " \
                "attention mechanism"
            assert coverage is not None, "lambda_coverage != 0.0 requires " \
                "coverage attention"

            shard_state.update({
                "std_attn": attns.get("std"),
                "coverage_attn": coverage
            })
        return shard_state

    def _compute_loss(self, batch, batch_idx, output, target, std_attn=None,
                      coverage_attn=None, step=0):
        # output: [tgt_len x batch x hidden]
        bottled_output = self._bottle(output) 

        scores = self.generator(bottled_output) # (len * batch, vocab)
        gtruth = target.view(-1) # (len * batch, )

        if hasattr(batch, "tgt_concept_index") and isinstance(self.criterion, KGLabelSmoothingLoss):
            concept_indices = batch.tgt_concept_index.repeat(output.shape[0], 1) # (len * batch, topk)
            loss = self.criterion(scores, gtruth, concept_indices, batch_idx)
        else:
            loss = self.criterion(scores, gtruth)

        # bag of concept loss
        if self.coefficient > 0 and (hasattr(batch, "tgt_concept_emb") or hasattr(batch, "tgt_concept_words_emb")):
            weight = min(self.coefficient, self.step_size * batch_idx)
            # print(self.coefficient, self.step_size, batch_idx, weight)
            # weight = self.coefficient
            sent_emb = output.mean(dim=0) # (batch, hidden)
            if hasattr(batch, "tgt_concept_emb"):
                concept_emb = batch.tgt_concept_emb # (batch, d_model)
                boc_loss = self.boc_criterion(sent_emb, concept_emb)
            if hasattr(batch, "tgt_concept_words_emb"):
                concept_emb = batch.tgt_concept_words_emb # (batch, d_model)
                boc_loss = self.boc_criterion(sent_emb, concept_emb)

                # probs = torch.exp(scores) * self.
                # total_embedding = 
            
            loss = loss + weight*boc_loss
            if random.random() < 0.005:
                print(weight, boc_loss.item(), loss.item())

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(
                std_attn=std_attn, coverage_attn=coverage_attn)
            loss += coverage_loss
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats

    def _compute_coverage_loss(self, std_attn, coverage_attn):
        covloss = torch.min(std_attn, coverage_attn).sum()
        covloss *= self.lambda_coverage
        return covloss


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
