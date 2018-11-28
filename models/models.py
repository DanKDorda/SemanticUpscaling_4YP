### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch

def create_model(opt):
    if opt.model == 'labelup':
        from .labelup_model import LabelUpModel
        if opt.isTrain:
            model = LabelUpModel()
        else:
            raise ValueError('testing mode not implemented yet')
            # model = InferenceModel()
    else:
    	# from .ui_model import UIModel
        raise ValueError('weird other mode also not implemented, choose different model option')
    	# model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
