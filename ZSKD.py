import os
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import data_info


class ZSKD():
    def __init__(self, dataset, teacher, num_sample, beta, t, batch_size, lr, iters, kl):
        self.dataset = dataset
        self.cwh, self.num_classes, self.student = data_info(self.dataset)
        self.teacher = teacher
        self.num_sample = num_sample
        self.beta = beta
        self.t = t
        self.batch_size = batch_size
        self.lr = lr
        self.iters = iters
        self.kl = kl

        self.gen_num = 1

    def build(self):

        # lim_0, lim_1 = 2, 2
        file_num = np.zeros((self.num_classes), dtype=int)

        def get_class_similarity():

            # Find last layer
            t_layer = list(self.teacher.children())[-1]
            while 'Sequential' in str(t_layer):
                t_layer = list(t_layer.children())[-1]

            # size(#class number, #weights in final-layer )
            t_weights = list(t_layer.parameters())[0].cuda()
            # Compute concentration parameter
            t_weights_norm = F.normalize(t_weights, p=2, dim=1)
            cls_sim = torch.matmul(t_weights_norm, t_weights_norm.T)
            cls_sim_norm = torch.div(cls_sim - torch.min(cls_sim),
                                     torch.max(cls_sim) - torch.min(cls_sim))
            return cls_sim_norm

        cls_sim_norm = get_class_similarity()

        print('\n'+'-'*30+' ZSKD start '+'-'*30)

        # generate synthesized images
        for k in range(self.num_classes):

            for b in self.beta:
                for _ in range(self.num_sample // len(self.beta) // self.batch_size // self.num_classes):

                    # sampling target label from Dirichlet distribution
                    dir_dist = torch.distributions.dirichlet.Dirichlet(
                        b * cls_sim_norm[k] + 0.0001)
                    y = Variable(dir_dist.rsample(
                        (self.batch_size,)), requires_grad=False)

                    # optimization for images
                    inputs = torch.randn(
                        (self.batch_size, self.cwh[0], self.cwh[1], self.cwh[2])).cuda()
                    inputs = Variable(inputs, requires_grad=True)
                    optimizer = torch.optim.Adam([inputs], self.lr)

                    for n_iter in range(self.iters):
                        optimizer.zero_grad()
                        logit = self.teacher(inputs) / 20.0
                        if self.kl:
                            l = 20 **2 * F.kl_div(F.log_softmax(logit, dim=1), y.detach(), size_average=False) / y.size(0)
                        else:
                            l = -torch.sum(F.log_softmax(logit / 20.0,
                                        dim=1) * y.detach()) / y.size(0)
                        l.backward()
                        optimizer.step()
                        if n_iter % 100 == 0:
                            print(f'\t[{n_iter}/{self.iters}] Loss: {l} ')

                    # save the synthesized images
                    t_cls = torch.argmax(y, dim=1).detach().cpu().numpy()
                    save_root = './saved_img/'+self.dataset+'/'
                    for m in range(self.batch_size):
                        save_dir = save_root+str(t_cls[m])+'/'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        if self.dataset == 'mnist':
                            vutils.save_image(inputs[m, :, :, :].data.clone(
                            ), save_dir + str(file_num[t_cls[m]]) + '.jpg')
                        else:
                            vutils.save_image(inputs[m, :, :, :].data.clone(
                            ), save_dir + str(file_num[t_cls[m]]) + '.jpg', normalize=True)

                        file_num[t_cls[m]] += 1
                    print('Generate {} synthesized images [{}/{}]'.format(
                        self.batch_size, self.batch_size*self.gen_num, self.num_sample))

                    self.gen_num += 1

        print('\n'+'-'*30+' ZSKD end '+'-'*30)

        return self.student, save_root
