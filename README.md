#### This repo contains the code for our work published in AAAI 2021: 

`CARE: Commonsense-Aware Emotional Response Generation with Latent Concepts`

The paper is available [here](https://arxiv.org/abs/2012.08377)

### Repo Structure:
1. ./emotion_classifier: our deepmoji classifier (https://github.com/huggingface/torchMoji).
2. ./process_data: scripts to process data, extract emotional triplets, and construct latent concepts
3. ./tools: scripts to expand VAD lexicon, process data, evaluate responses, and do helper tasks
4. ./OpenKE: scripts to train TransE on ConceptNet, based on (https://github.com/thunlp/OpenKE)
5. ./onmt: the main package for our model, based on (https://github.com/OpenNMT/OpenNMT-py)

### Steps:
1. Train an emotion classifier on emotional tweets
2. Download and preprocess dataset
3. Use the trained emotion classifer to classify the emotions of responses
4. Extract emotional triplets using scripts in ./process_data using PMI
5. Extend ConceptNet with extracted emotional triplets
6. Train a TransE model on the extended ConceptNet using ./OpenKE
7. Extract concepts in messages
8. Generate relational and emotional latent concepts for the responses
9. run run_reddit.sh/run_twitter.sh to train and evaluate the model

### Datasets
- Reddit: https://www.dropbox.com/s/p429s2yhzkti6ra/reddit.zip?dl=0
- Twitter: https://www.dropbox.com/s/u6grauxtv8xptqw/twitter.zip?dl=0

The transformer is based on the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) package (v1.0.0). Follow their doc for the preprocessing and vocabulary construction of the conversation datasets.
The training, generation, and evaluation follow the shell scripts.
