import numpy as np
import torch
from torch.autograd import Variable

from .base_model import BaseModel
from util.image_pool import ImagePool
from . import networks


class LabelUpModel(BaseModel):

    def name(self):
        return 'LabelUpModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, d_real_oc, d_fake_oc, loss_G_GLU):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake, d_real_oc, d_fake_oc, loss_G_GLU), flags) if f]

        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        input_set = None

        ##### define network
        # generators
        # net that makes 35 shitty layers into 35 glorious hi-def layers
        self.netG_multichan = networks.define_G(input_nc, input_nc, opt.num_phases, gpu_ids=self.gpu_ids,
                                                res_blocks=opt.num_res_blocks)
        # net that takes those 35 glorious layers and sticks 'em all together into a legit semantic map
        self.netG_glue = networks.define_GlueNet(input_nc, input_set, gpu_ids=self.gpu_ids)

        # discriminator
        if self.isTrain:
            self.netD = networks.define_D(input_nc, 64, 3, gpu_ids=self.gpu_ids)  # i feel like this should be 35...
            self.netD_glue = networks.define_D(1, 64, 3, gpu_ids=self.gpu_ids)

        # encoder for reading features, mite be useful
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks -> inside G there numPhases blocks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG_multichan, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:

            # losses
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'D_real_OC', 'D_fake_OC', 'loss_G_GLU')

            # optimizer G -> not sure what the first bit does, but apparently not all of the generator is trained at once
            if opt.niter_fix_global > 0:
                finetune_list = set()
                params_dict = dict(self.netG_multichan.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print(
                    '------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG_multichan.models.parameters())
                # params = list([sub_model.parameters() for model in self.netG.models for sub_model in model])
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            #optimizer G_glue
            params = list(self.netG_glue.models.parameters())
            self.optimizer_G_glue = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            #optimizer D_glue
            params = list(self.netD_glue.parameters())
            self.optimizer_D_glue = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, net, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return net.forward(fake_query)
        else:
            return net.forward(input_concat)

    # used in training
    def forward(self, low_res, inst, high_res, feat, phase, blend, infer=False):
        # encode inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(low_res, inst, high_res, feat)

        #### fake generation
        # this here so that eventually we'll use feat
        input_concat = input_label

        # output is (batch, 35, resolution), dtype float32
        fake_onehots = self.netG_multichan.forward(input_concat, phase, blend)

        # output is (batch, 1, resolution), with values of the semantic elements
        fake_one_layer = self.netG_glue(fake_onehots)

        # NEW LOSS
        # there's loss for the 35 layer dingo, to tell it apart from the one hot high res dingo
        # there's loss for the final 1 layer image, to tell it apart from the final image
        # these are both discriminators for the fakes
        # so we have two discriminators now yeaa!
        #HMM but we're using the same fake pool so idk what's up with that

        ### fake detection and loss
        # its multi_layer_bro
        pred_fake_pool = self.discriminate(input_label, fake_onehots, self.netD, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        # it's single layer ho
        pred_fake_pool_onechan = self.netD_glue.forward(torch.cat(low_res, fake_one_layer))
        loss_D_fake_onechan = self.criterionGAN(pred_fake_pool_onechan, False)

        # Real Detection and Loss
        # do I need two here? I'm not really sure, but fuck it why not
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)
        #onechan mode
        pred_real_onechan = self.netD_glue.forward(torch.cat(input_label, real_image))
        loss_D_real_onechan = self.criterionGAN(pred_real_onechan, True)


        # more loss -> GAN passability, VGG, Label match

        # GAN loss do we even need this seperate fucker here tho?????????
        pred_fake = self.netD.forward(torch.cat((input_label, fake_onehots), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN loss for the GlueGAN
        pred_fake = self.netD_glue.forward(torch.cat(low_res, fake_one_layer), dim=1)
        loss_G_GLU = self.criterionGAN(pred_fake, True)
        loss_G_GAN_Feat = 0

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            # hack together three channel input
            loss_G_VGG = self.criterionVGG(torch.cat((fake_one_layer, fake_one_layer, fake_one_layer), dim=1),
                                           torch.cat((real_image, real_image, real_image),
                                                     dim=1)) * self.opt.lambda_feat

        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_D_real_onechan, loss_D_fake_onechan, loss_G_GLU),
                None if not infer else fake_onehots]

    # used in testing, returns only fake images
    def inference(self):
        pass

    ### COPIED METHODS FOR EXTRACTING EDGES AND NETWORK TRAINING UTILS
    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG_multichan.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
