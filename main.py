import os
from glob import glob
from typing import Optional

import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator
import torch
import torchvision
import torch.nn as nn
import torch
import torch.nn as nn
from pretrainedmodels import inceptionresnetv2
from torchsummary import summary
import torch.nn.functional as F
from torch.jit import ScriptModule, script_method, trace
import functools
from torchsummary import summary


with open('config/config.yaml') as cfg:
    config = yaml.load(cfg, Loader=yaml.FullLoader)
model = get_generator(config['model'])
weights_path = 'fpn_inception.h5'
model.load_state_dict(torch.load(weights_path)['model'])

model.eval()
model = model.cpu()
# summary(model, input_size=(3, 256, 256))

# model = model.cuda()
model.train(True)
#model = model.cpu()
example = torch.rand(1, 3, 128, 128)# .to('cuda')
traced_script_module = torch.jit.trace(model.module, example)
traced_script_module.save("model128.pt")

