import os
import time
import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import gc

data_dir = "../data/KB/ConceptNet-KE/reddit_weight_1.0_freq_5/"
ckpt_dir = "/boot/data/KE-checkpoint/"
config_id = int(time.time())
# config_id = 1574163237
ckpt_fname = "RotatE{0}.ckpt".format(config_id)
ckpt_path = os.path.join(ckpt_dir, ckpt_fname)

# best params: 100, 6, 8, 1024
# best params: 100, 6, 16, 1024:  0.8717

params = {
    "embed_dim": [200],
    "margin": [3,6,10],
    "negative_samples": [4,8,16],
    "batch_size": [512,1024,2048],
    "alpha": [0.001]
}

# hyper-parameter tunings
embed_dim = 200 # 100, 200, 400
train_times = 10

for margin in params["margin"]:
    for negative_samples in params["negative_samples"]:
        for batch_size in params["batch_size"]:
            for alpha in params["alpha"]:
                print(embed_dim, margin, negative_samples, batch_size, alpha, train_times)

                # dataloader for training
                train_dataloader = TrainDataLoader(
                    in_path = data_dir, 
                    batch_size = batch_size,
                    threads = 8,
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
                trainer = Trainer(model = model, data_loader = train_dataloader, train_times = train_times, alpha = alpha, use_gpu = True, opt_method = "adam")
                trainer.run()
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

                del train_dataloader
                del test_dataloader
                del rotate
                del model
                del trainer
                del tester
                gc.collect()
