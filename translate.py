#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus, split_emotions
from onmt.translate.translator import build_translator
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from tools.util import load_pickle

def read_emotion_file(path):
    with open(path, "r") as f:
        data = [int(line.strip("\n")) for line in f.readlines()]
    return data

def translate(opt): 
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    # shard_pairs = zip(src_shards, tgt_shards)
    # print("number of shards: ", len(src_shards), len(tgt_shards))

    # load emotions
    tgt_emotion_shards = [None]*100
    if opt.target_emotions_path != "":
        print("Loading target emotions...")
        tgt_emotions = read_emotion_file(opt.target_emotions_path)
        tgt_emotion_shards = split_emotions(tgt_emotions, opt.shard_size)
        # print("number of shards: ", len(tgt_emotion_shards))
    
    tgt_concept_embedding_shards = [None]*100
    if opt.target_concept_embedding != "":
        print("Loading target_concept_embedding...")
        tgt_concept_embedding = load_pickle(opt.target_concept_embedding)
        tgt_concept_embedding_shards = split_emotions(tgt_concept_embedding, opt.shard_size)
        # print("number of shards: ", len(tgt_concept_embedding_shards))
    
    tgt_concept_words_shards = [None]*100
    if opt.target_concept_words != "":
        print("Loading target_concept_words...")
        tgt_concept_words = load_pickle(opt.target_concept_words)
        # tgt_concept_words_shards = split_emotions(zip(tgt_concept_words), opt.shard_size)
        tgt_concept_words_shards = [tgt_concept_words]
        # print("number of shards: ", len(tgt_concept_words_shards))
    
    shard_pairs = zip(src_shards, tgt_shards, tgt_emotion_shards, tgt_concept_embedding_shards, tgt_concept_words_shards)

    for i, (src_shard, tgt_shard, tgt_emotion_shard, tgt_concept_embedding_shard, tgt_concept_words_shard) in enumerate(shard_pairs):
        # print(len(src_shard), len(tgt_shard), len(tgt_emotion_shard))
        logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            tgt_emotion_shard=tgt_emotion_shard,
            rerank=opt.rerank,
            emotion_lexicon=opt.emotion_lexicon,
            tgt_concept_embedding_shard=tgt_concept_embedding_shard,
            tgt_concept_words_shard=tgt_concept_words_shard
            )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
