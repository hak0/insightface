#/bin/python3

import torch

path = "work_dirs/ms1mv3_mobileface_lr02/lx_other_info_0.pth.tar"
d = torch.load(path)
# for batchsize=256 and ms1mbf dataset
# global_step = epoch * 20230 + 10
d["epoch"] = 30
d["global_step"] = 606910 # for bach_size=256
torch.save(d, path)
