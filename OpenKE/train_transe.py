import os
import time
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

data_dir = "../data/KB/ConceptNet-KE/twitter_weight_1.0_freq_5/"
# data_dir = "../data/KB/ConceptNet-KE/reddit_weight_1.0_freq_5/"
ckpt_dir = "/boot/data/KE-checkpoint/"
config_id = int(time.time())
# config_id = 1574163237
ckpt_fname = "TransE{0}.ckpt".format(config_id)
ckpt_path = os.path.join(ckpt_dir, ckpt_fname)

# hyper-parameter tunings
embed_dim = 100 
margin = 0.4
negative_samples = 4
nbatches = 100
alpha = 0.001
train_times = 800

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = data_dir, 
	nbatches = nbatches,
	threads = 8, 
	sampling_mode = "cross", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = negative_samples,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader(data_dir, "triple")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = embed_dim, 
	p_norm = 2, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = margin),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, opt_method = "adam", train_times = train_times, \
	alpha = alpha, use_gpu = True, checkpoint_dir=ckpt_path, save_steps=100)
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
trainer.run(tester, test_every=100)
print("Saving model to {0}...".format(ckpt_path))
transe.save_checkpoint(ckpt_path)

# test the model
print("Testing...")
print("Loading model from {}...".format(ckpt_path))
transe.load_checkpoint(ckpt_path)
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
acc, threshlod = tester.run_triple_classification()
print("accuracy: ", acc)