import os
import time
import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

data_dir = "../data/KB/ConceptNet-KE/reddit_weight_1.0_freq_5/"
ckpt_dir = "/boot/data/KE-checkpoint/"
config_id = int(time.time())
# config_id = 1574163237
ckpt_fname = "RotatE{0}.ckpt".format(config_id)
ckpt_path = os.path.join(ckpt_dir, ckpt_fname)

# hyper-parameter tunings
embed_dim = 100 # 100, 200, 300
margin = 6 # 1, 3, 6, 10, 15, 20
negative_samples = 16 # 64, 128
batch_size = 4092 # 512, 1024, 2048
alpha = 0.001 # 0.001,0.01
train_times = 400

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path = data_dir, 
    batch_size = batch_size,
    threads = 2,
    sampling_mode = "cross", 
    bern_flag = 0, 
    filter_flag = 1, 
    neg_ent = negative_samples,
    neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader(data_dir, "triple")
# test_dataloader = TestDataLoader(data_dir, "link")


# define the model
rotate = RotatE(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = embed_dim,
    margin = margin,
    epsilon = 2.0,
)

# define the loss function
model = NegativeSampling(
    model = rotate, 
    loss = SigmoidLoss(adv_temperature = 2),
    batch_size = train_dataloader.get_batch_size(), 
    regul_rate = 0.0
)

# train the model
print("Training...")
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = train_times, alpha = alpha, use_gpu = True, 
    opt_method = "adam", checkpoint_dir=ckpt_dir, save_steps=50)
tester = Tester(model = rotate, data_loader = test_dataloader, use_gpu = True)
trainer.run(tester, test_every=25)
# print("Saving model to {0}...".format(ckpt_path))
# rotate.save_checkpoint(ckpt_path)

# test the model
print("Testing...")
# print("Loading model from {}...".format(ckpt_path))
# rotate.load_checkpoint(ckpt_path)
tester = Tester(model = rotate, data_loader = test_dataloader, use_gpu = True)
# tester.run_link_prediction(type_constrain = False)
acc, threshlod = tester.run_triple_classification()
print("accuracy: ", acc)