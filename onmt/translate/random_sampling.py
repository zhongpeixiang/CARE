import torch
import torch.nn.functional as F

from onmt.translate.decode_strategy import DecodeStrategy


def sample_with_temperature(logits, sampling_temp, keep_topk, \
        emotion_topk_decoding_temp=0, tgt_concept_words=None, \
        score_temp=None, VAD_score_temp=None, memory_bank=None, src_relevance_temp=None, VAD_lambda=None, \
        decoder_word_embedding=None):
    """Select next tokens randomly from the top k possible next tokens.

    Samples from a categorical distribution over the ``keep_topk`` words using
    the category probabilities ``logits / sampling_temp``.

    Args:
        logits (FloatTensor): Shaped ``(batch_size, vocab_size)``.
            These can be logits (``(-inf, inf)``) or log-probs (``(-inf, 0]``).
            (The distribution actually uses the log-probabilities
            ``logits - logits.logsumexp(-1)``, which equals the logits if
            they are log-probabilities summing to 1.)
        sampling_temp (float): Used to scale down logits. The higher the
            value, the more likely it is that a non-max word will be
            sampled.
        keep_topk (int): This many words could potentially be chosen. The
            other logits are set to have probability 0.

    Returns:
        (LongTensor, FloatTensor):

        * topk_ids: Shaped ``(batch_size, 1)``. These are
          the sampled word indices in the output vocab.
        * topk_scores: Shaped ``(batch_size, 1)``. These
          are essentially ``(logits / sampling_temp)[topk_ids]``.
    """

    if sampling_temp == 0.0 or keep_topk == 1:
        # For temp=0.0, take the argmax to avoid divide-by-zero errors.
        # keep_topk=1 is also equivalent to argmax.
        topk_scores, topk_ids = logits.topk(1, dim=-1)
        if sampling_temp > 0:
            topk_scores /= sampling_temp
    else:
        logits = torch.div(logits, sampling_temp)

        if keep_topk > 0:
            top_values, top_indices = torch.topk(logits, keep_topk, dim=1) # top_indices: (batch, topk)
            kth_best = top_values[:, -1].view([-1, 1])
            kth_best = kth_best.repeat([1, logits.shape[1]]).float() # (batch, vocab)

            # Set all logits that are not in the top-k to -10000.
            # This puts the probabilities close to 0.
            ignore = torch.lt(logits, kth_best)
            logits = logits.masked_fill(ignore, -10000)

            if emotion_topk_decoding_temp != 0 and tgt_concept_words is not None:
                # print(score_temp[tgt_concept_words[0]])
                # print(VAD_score_temp[tgt_concept_words[0]])
                # print(VAD_lambda[tgt_concept_words[0]])
                # update logits for concepts
                tgt_concept_words_emb = decoder_word_embedding.word_lut(tgt_concept_words[0]) # (batch, top_k, d_model)
                score_softmax = torch.softmax(score_temp[tgt_concept_words[0]] * tgt_concept_words[1], dim=-1) # (batch, top_k)
                VAD_score_softmax = torch.softmax(VAD_score_temp[tgt_concept_words[0]] * tgt_concept_words[2], dim=-1) # (batch, top_k)
                
                # src relevance
                # src_relevance = torch.softmax(src_relevance_temp[tgt_concept_words[0]] * torch.bmm(tgt_concept_words_emb, memory_bank.mean(dim=0).unsqueeze(-1)).squeeze(-1), dim=-1) # (batch, top_k)
                # score_softmax = (score_softmax + src_relevance)/2
                
                clamped_VAD_lambda = VAD_lambda.clamp(1e-4, 1-(1e-4))
                weight = score_softmax * clamped_VAD_lambda[tgt_concept_words[0]] + VAD_score_softmax * (1-clamped_VAD_lambda[tgt_concept_words[0]])
                # print(weight.shape)
                # print(weight)
                tgt_concept_words_emb = (weight.unsqueeze(-1) * tgt_concept_words_emb).sum(dim=1) # (batch, emb_dim)
                # print(tgt_concept_words_emb.shape)
                # print(top_indices.shape, tgt_concept_words_emb.shape)
                concept_scores = torch.bmm(decoder_word_embedding.word_lut(top_indices), tgt_concept_words_emb.unsqueeze(-1)).squeeze(-1) # (batch, top_k)
                # print(concept_scores.shape)
                # print(concept_scores)
                concept_logits = F.log_softmax(concept_scores, dim=-1) # (batch, top_k)
                # print((logits.gather(1, top_indices) + emotion_topk_decoding_temp*concept_logits).shape)
                # print(logits.gather(1, top_indices))
                # print(concept_logits)
                logits.scatter_(1, top_indices, logits.gather(1, top_indices) + emotion_topk_decoding_temp*concept_logits)
                # raise SystemExit()
                # logits[tgt_concept_words[0]] = logits[tgt_concept_words[0]] + emotion_topk_decoding_temp*torch.log(weight) # add two logits
                # new_logits = logits.gather(1, tgt_concept_words[0]) + emotion_topk_decoding_temp*torch.log(weight)
                # logits.scatter_(1, tgt_concept_words[0], logits.gather(1, tgt_concept_words[0]) + emotion_topk_decoding_temp*torch.log(weight))
        
        dist = torch.distributions.Multinomial(
            logits=logits, total_count=1)
        topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
        topk_scores = logits.gather(dim=1, index=topk_ids)
    return topk_ids, topk_scores


class RandomSampling(DecodeStrategy):
    """Select next tokens randomly from the top k possible next tokens.

    The ``scores`` attribute's lists are the score, after applying temperature,
    of the final prediction (either EOS or the final token in the event
    that ``max_length`` is reached)

    Args:
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        batch_size (int): See base.
        device (torch.device or str): See base ``device``.
        min_length (int): See base.
        max_length (int): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        return_attention (bool): See base.
        max_length (int): See base.
        sampling_temp (float): See
            :func:`~onmt.translate.random_sampling.sample_with_temperature()`.
        keep_topk (int): See
            :func:`~onmt.translate.random_sampling.sample_with_temperature()`.
        memory_length (LongTensor): Lengths of encodings. Used for
            masking attention.
    """

    def __init__(self, pad, bos, eos, batch_size, device,
                 min_length, block_ngram_repeat, exclusion_tokens,
                 return_attention, max_length, sampling_temp, keep_topk,
                 memory_length, emotion_topk_decoding_temp=0,
                 score_temp=None, VAD_score_temp=None, src_relevance_temp=None, 
                 VAD_lambda=None, decoder_word_embedding=None):
        super(RandomSampling, self).__init__(
            pad, bos, eos, batch_size, device, 1,
            min_length, block_ngram_repeat, exclusion_tokens,
            return_attention, max_length)
        self.sampling_temp = sampling_temp
        self.keep_topk = keep_topk
        self.topk_scores = None
        self.memory_length = memory_length
        self.batch_size = batch_size
        self.select_indices = torch.arange(self.batch_size,
                                           dtype=torch.long, device=device)
        self.original_batch_idx = torch.arange(self.batch_size,
                                               dtype=torch.long, device=device)
        self.emotion_topk_decoding_temp = emotion_topk_decoding_temp
        self.score_temp = score_temp
        self.VAD_score_temp = VAD_score_temp
        self.src_relevance_temp = src_relevance_temp
        self.VAD_lambda = VAD_lambda
        self.decoder_word_embedding = decoder_word_embedding

    def advance(self, log_probs, attn, tgt_concept_words=None, memory_bank=None):
        """Select next tokens randomly from the top k possible next tokens.

        Args:
            log_probs (FloatTensor): Shaped ``(batch_size, vocab_size)``.
                These can be logits (``(-inf, inf)``) or log-probs
                (``(-inf, 0]``). (The distribution actually uses the
                log-probabilities ``logits - logits.logsumexp(-1)``,
                which equals the logits if they are log-probabilities summing
                to 1.)
            attn (FloatTensor): Shaped ``(1, B, inp_seq_len)``.
            tgt_concept_words: tuple of (batch_size, top_k_concepts)
        """

        self.ensure_min_length(log_probs)
        self.block_ngram_repeats(log_probs)
        topk_ids, self.topk_scores = sample_with_temperature(
            log_probs, self.sampling_temp, self.keep_topk, 
            self.emotion_topk_decoding_temp, tgt_concept_words, 
            self.score_temp, self.VAD_score_temp, memory_bank, self.src_relevance_temp, 
            self.VAD_lambda, self.decoder_word_embedding)

        self.is_finished = topk_ids.eq(self.eos)

        self.alive_seq = torch.cat([self.alive_seq, topk_ids], -1)
        if self.return_attention:
            if self.alive_attn is None:
                self.alive_attn = attn
            else:
                self.alive_attn = torch.cat([self.alive_attn, attn], 0)
        self.ensure_max_length()

    def update_finished(self):
        """Finalize scores and predictions."""
        # shape: (sum(~ self.is_finished), 1)
        finished_batches = self.is_finished.view(-1).nonzero()
        for b in finished_batches.view(-1):
            b_orig = self.original_batch_idx[b]
            self.scores[b_orig].append(self.topk_scores[b, 0])
            self.predictions[b_orig].append(self.alive_seq[b, 1:])
            self.attention[b_orig].append(
                self.alive_attn[:, b, :self.memory_length[b]]
                if self.alive_attn is not None else [])
        self.done = self.is_finished.all()
        if self.done:
            return
        is_alive = ~self.is_finished.view(-1)
        self.alive_seq = self.alive_seq[is_alive]
        if self.alive_attn is not None:
            self.alive_attn = self.alive_attn[:, is_alive]
        self.select_indices = is_alive.nonzero().view(-1)
        self.original_batch_idx = self.original_batch_idx[is_alive]
