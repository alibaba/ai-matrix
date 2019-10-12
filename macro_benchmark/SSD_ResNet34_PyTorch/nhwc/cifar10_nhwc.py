# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import pylab
import numpy as np
from torch.autograd import Variable

from argparse import ArgumentParser
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel as DDP

from vgg_nhwc import VggBN_NHWC
from resnet_nhwc_cifar10 import *

def parse_args():
    parser = ArgumentParser(description="NHWC CIFAR10 test")
    parser.add_argument('--local_rank', default=0, type=int,
			help='Used for multi-process training. Can either be manually set ' +
			'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument('--net', type=str, choices=['vgg', 'rn18', 'rn34'], default='vgg')

    return parser.parse_args()

def train_and_test(args):
    # Setup multi-GPU if necessary
    args.distributed = False
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])

    if args.distributed:
        torch.cuda.set_device(args.local_rank)

        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')


    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    batch_size = 32
    learning_rate = 0.01
    learning_rate_decay = 0.0005
    momentum = 0.9
    epoch_step = 25
    max_epoch = 300

    transform_train = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    else:
        train_sampler = None
        val_sampler = None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=train_sampler is None,
                                              sampler=train_sampler, num_workers=4)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=val_sampler is None,
                                             sampler=val_sampler, num_workers=4)

    save_dir = "./save"
    if not os.path.exists(save_dir) and args.local_rank == 0:
        os.mkdir(save_dir)


    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    if args.net == 'vgg':
        net = VggBN_NHWC()
    elif args.net == 'rn18':
        net = ResNet18_NHWC()
    elif args.net == 'rn34':
        net = ResNet34_NHWC()

    net = network_to_half(net.cuda())
    net.apply(weights_init)
    net.cuda()


    criterion = nn.CrossEntropyLoss()
    base_lr = learning_rate * args.world_size
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=learning_rate_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epoch_step, gamma=0.5)
    optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.)

    if args.distributed:
        net = DDP(net)

    test_accuracies = np.zeros(max_epoch)
    for epoch in range(max_epoch):  # loop over the dataset multiple times
        #pbar = tqdm(trainloader)
        #pbar.mininterval = 1 # update the processing bar at least 1 second

        """
            Initial Check
        """
        """
        net.eval()

        if epoch == 0:
            print('\033[0;31mInitial Check: \033[0m')
            running_loss, correct, total = 0., 0., 0.
            for i, data in enumerate(testloader, 0):
                images, labels = data
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
                images = images.half().permute(0, 2, 3, 1).contiguous()
                outputs = net(images)
                loss = criterion(outputs, labels)
                running_loss = running_loss * (i/(i+1.)) + loss.data[0] * (1./(i+1.) )
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.data).sum()
            print('Loss on the test images: %f  ......  should be 2.3' % running_loss)
            print('Accuracy on the test images: %f %% ......  should be 10%%' % (100. * correct / total))
        """

        """
            Training ...
        """
        net.train()

        running_loss, correct, total = 0., 0., 0.
        scheduler.step()

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            inputs = inputs.half().permute(0, 2, 3, 1).contiguous()
            inputs.require_grad = False
            labels.require_grad = False
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.backward(loss)
            optimizer.step()

            # update statistics
            running_loss = running_loss * (i/(i+1.)) + loss.data[0] * (1./(i+1.) )
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        print('\033[0;32m Statistics on epoch :%d learning rate: %f\033[0m' %(epoch, scheduler.get_lr()[0]))
        print('Train Loss : %f Train Accuracy: %f %%' % (running_loss, 100. * correct / total))

        """
            Testing ...
        """
        net.eval()

        correct, total = 0., 0.
        for data in testloader:
            images, labels = data
            images = Variable(images.cuda()).half().permute(0, 2, 3, 1).contiguous()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()

        print('[{}] {} / {} correct'.format(args.local_rank, correct, total))
        print('Test Accuracy: \033[1;33m%f %%\033[0m' % (100. * correct / total))
        test_accuracies[epoch] = 100. * correct / total

        """
            Saving model and accuracies, and ploting
        """
        #np.save('./save/accuracies.npy', test_accuracies)
        #torch.save(net.state_dict(), './save/model.%d.pkl' %epoch)

        #plt.figure()
        #pylab.xlim(0, max_epoch + 1)
        #pylab.ylim(0, 100)
        #plt.plot(range(1, max_epoch +1), test_accuracies)
        #plt.savefig('./save/accuracies.png')
        #plt.close()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = parse_args()

    train_and_test(args)


