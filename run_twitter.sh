#!/bin/sh

seed=1234
dataset="twitter"
tgt_concept_words_type=3
word_vec_size=300
layers=1
heads=4
transformer_ff=512
dropout=0.1
word_embedding_path="./data/Twitter/twitter-glove-300d.pt"
optim="adam"
decay_method="noam"
learning_rate=1
label_smoothing=0.1
batch_size=64
train_steps=80000
warmup_steps=6000
translate_step=$train_steps
topk=10
emotion_topk_decoding_temp=1

translation_src="valid-src"
translation_tgt="valid-tgt"
train_emotions="train-tgt-emotions"
valid_emotions="valid-tgt-emotions"
train_tgt_concept_words="train-tgt-concept-topk-words-temp-40-cleaned" 
valid_tgt_concept_words="valid-tgt-concept-topk-words-temp-40-cleaned" 
translation_tgt_concept_words="valid-tgt-input-concept-topk-words-temp-40-cleaned" # use in translation

cmd="python"
config_id=$(date +%s)
# config_id=1575608609
echo $config_id


$cmd train.py --config_id $config_id --data ./data/Twitter/$dataset --word_vec_size $word_vec_size \
--encoder_type kg_transformer --decoder_type kg_transformer --position_encoding --share_decoder_embeddings \
--pre_word_vecs_enc $word_embedding_path --pre_word_vecs_dec $word_embedding_path \
--layers $layers --heads $heads --transformer_ff $transformer_ff --rnn_size $word_vec_size --dropout $dropout --batch_size $batch_size \
--train_steps $train_steps --optim $optim --decay_method $decay_method --warmup_steps $warmup_steps --learning_rate $learning_rate --label_smoothing $label_smoothing --param_init_glorot \
--save_model ./saved_model/${dataset}_${config_id} --log_file ./log/training.txt --gpu_ranks 0 --seed $seed \
--train_tgt_concept_words ./data/Twitter/${train_tgt_concept_words}.pkl --valid_tgt_concept_words ./data/Twitter/${valid_tgt_concept_words}.pkl \
--num_emotion_classes 6 --emotion_emb_size 50 --train_emotion_path ./data/Twitter/${train_emotions}.txt --valid_emotion_path ./data/Twitter/${valid_emotions}.txt \
--tgt_concept_words_type $tgt_concept_words_type \


$cmd translate.py --model ./saved_model/${dataset}_${config_id}_step_${translate_step}.pt \
--src ./data/Twitter/${translation_src}.txt --tgt ./data/Twitter/${translation_tgt}.txt \
--output ./outputs/${dataset}_${config_id}_step_${translate_step}_temp_${emotion_topk_decoding_temp}.txt \
--fp32 --gpu 0 --share_vocab --random_sampling_topk $topk --beam_size 1 --seed $seed --emotion_topk_decoding_temp $emotion_topk_decoding_temp \
--target_concept_words ./data/Twitter/${translation_tgt_concept_words}.pkl \
--target_emotions_path ./data/Twitter/${translation_src}-input-emotions.txt \

# replace emnpt generated response by .
$cmd ./tools/clean_generated_text.py --path ./outputs/${dataset}_${config_id}_step_${translate_step}_temp_${emotion_topk_decoding_temp}.txt


$cmd evaluate.py --config_id $config_id --translate_step $translate_step --dataset $dataset \
--target_file ./data/Twitter/${translation_tgt}.txt --output_file ./outputs/${dataset}_${config_id}_step_${translate_step}_temp_${emotion_topk_decoding_temp}.txt


# compute emotion accuracy
$cmd ./emotion_classifier/create_torchmoji_embedding.py --path ./outputs/${dataset}_${config_id}_step_${translate_step}_temp_${emotion_topk_decoding_temp}.txt
$cmd ./emotion_classifier/classify_emotion.py --path ./outputs/${dataset}_${config_id}_step_${translate_step}_temp_${emotion_topk_decoding_temp}-torchmoji.npy
$cmd ./tools/compute_emotion_metrics.py \
--output_emotions_path ./outputs/${dataset}_${config_id}_step_${translate_step}_temp_${emotion_topk_decoding_temp}-emotions.txt \
--target_emotions_path ./data/Twitter/${translation_src}-input-emotions.txt
