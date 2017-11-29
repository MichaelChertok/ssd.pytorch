import os,sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from tensorboard_logger import configure, log_value, Logger


# get the resnet 101.


import numpy as np
import time

from os.path import expanduser
homeDir = expanduser('~')
sys.path.append(os.path.join(homeDir, 'YellowFin_Pytorch/tuner_utils/'))
from yellowfin import YFOptimizer

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")



parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--accum_batch_size', default=32, type=int, help='Effective batch size for training (accumulate gradients over this number of images')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--checkpoint_freq', default=5000, type=int, help='checkpoint saving frequency (iterations)')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('--use_hint', default=False, type=str2bool, help='Use Network Priming')
parser.add_argument('--hint_vgg', default=False, type=str2bool, help='Apply priming to vgg stage')
parser.add_argument('--hint_extra', default=False, type=str2bool, help='Apply priming to extra layers stage')
parser.add_argument('--hint_loc', default=False, type=str2bool, help='Apply priming to localization layers stage')
parser.add_argument('--hint_conf', default=False, type=str2bool, help='Apply priming to classifications layers stage')
parser.add_argument('--controller_bias', default=False, type=str2bool, help='Learn bias factor in controllers')
parser.add_argument('--controller_sigmoid', default=False, type=str2bool, help='Use sigmoid before applying controllers')
parser.add_argument('--optimizer',default='sgd',help='Optimizer type. Currently supported: Adam or SGD. lr argument is ignored for Adam')
parser.add_argument('--experiment_name',default='my_exp',help='name of experiment,used for tensorboard logging')
parser.add_argument('--insert_bn_layers',default=False,type=str2bool,help='resinert BN layers before training')
parser.add_argument('--allow_new_weights',default=False,type=str2bool,help='allow new default weights when loading network with missing dictionary values')
parser.add_argument('--residual_controllers',default=False,type=str2bool,help='use residual controllers')
parser.add_argument('--hard_initialization',default=False,type=str2bool,help='deterministic initialization of controllers')
parser.add_argument('--add_relu',default=False,type=str2bool,help='apply ReLu after priming layer')
parser.add_argument('--train_on',default='train',help='train on the validation set')
parser.add_argument('--n_to_learn',default='100_100_100_100',help='how many priming layer to actually learn in each network segment')
parser.add_argument('--hint_constants',default='1_0',help='set value of positive/negative hinted classes to pos_neg')
parser.add_argument('--hint_interpreter',default=False,type=str2bool,help='add extra layer after hint')
parser.add_argument('--year',default='2012',help='pascal year to train on')
parser.add_argument('--top-down-source',default='',help='filename for datasource of top-down image data (e.g, resnet results')

args = parser.parse_args()
n_to_learn = [int(q) for q in args.n_to_learn.split('_')]
hint_constants = [float(q) for q in args.hint_constants.split('_')]
if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = (v1, v2)[args.version == 'v2']

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

#train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
train_sets = [(args.year, args.train_on)] # remember you changed this....
#if args.train_on_val:
#    train_sets = [('2007', 'val')]

# train_sets = 'train'
ssd_dim = 300  # only support 300 now
means = (104, 117, 123)  # only support voc now
num_classes = len(VOC_CLASSES) + 1
batch_size = args.batch_size
accum_batch_size = args.accum_batch_size
iter_size = accum_batch_size / batch_size
max_iter = args.iterations
weight_decay = 0.0005
stepvalues = (80000, 100000, 120000)
gamma = args.gamma
momentum = args.momentum

if args.visdom:
    import visdom
    viz = visdom.Visdom()
if args.top_down_source != '':
    args.hint_interpreter=True
ssd_net = build_ssd('train', 300, num_classes, controller_bias=args.controller_bias, controller_sigmoid=args.controller_sigmoid,
                    insertBNLayers=args.insert_bn_layers, residual_controllers= args.residual_controllers,
                    add_relu=args.add_relu,n_to_learn=n_to_learn,add_hint_interpreter=args.hint_interpreter)
net = ssd_net


if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

import itertools
if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))

    oldNetDict = torch.load(args.resume)
    new_net_dict = ssd_net.state_dict ()
    for k in oldNetDict.keys():
        if k in new_net_dict:
            new_net_dict[k] = oldNetDict[k]
    ssd_net.load_state_dict(new_net_dict)

    #ssd_net.load_weights(args.resume)

    hinting_params = [ssd_net.vgg_controllers.parameters(),
                      ssd_net.extra_controllers.parameters(),
                      ssd_net.loc_controllers.parameters(),
                      ssd_net.conf_controllers.parameters()]
    if args.use_hint:
        print('Freezing all parameters except for hinting params.')
        for p in ssd_net.parameters():
            if type(p) is not nn.BatchNorm2d: # Disable batchnorm learning (probably no effect, there's none by default)
                p.requires_grad = False
            #else:
            #    p.requires_grad = False
        

        hinting_values = [args.hint_vgg,args.hint_extra,args.hint_loc,args.hint_conf]
        for use_my_hints,params in zip(hinting_values,hinting_params):
            if use_my_hints:
                for p in params:
                    p.requires_grad=True
    else:
        for pp in hinting_params:
            for p in pp:
                p.requires_grad=False
    if args.hard_initialization:
        a = ssd_net.state_dict()
        use_bias = args.controller_bias
        #for k, v in a.iteritems():  # keys():
        for k in a.keys():  # keys():
            if 'control' in k:
                print(k)
                v = a[k]
                if args.residual_controllers:
                    v[:] = 0 # set all to 0,because the residual will take care of this.
                else:
                    # break
                    if use_bias:
                        s = int(v.size()[0])//2
                        v[:s, :] = 1
                        v[s:, :] = 0
                    else:
                        v[:] = 1
        ssd_net.load_state_dict(a)

else:
    assert not args.use_hint,'Not supporting learning to prime for a non-pretrained model'
    vgg_weights = torch.load(args.basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)



if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

if args.cuda:
    net = net.cuda()


logger = Logger('runs/' + args.experiment_name)

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

curParams = [p for p in net.parameters() if p.requires_grad]
if args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(curParams, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer.lower()=='adam':
    #args.lr=1e-3
    optimizer = optim.Adam(params=curParams)
elif args.optimizer.lower()=='rmsprop':
    #args.lr=1e-2
    optimizer = optim.RMSprop(params=curParams)
elif args.optimizer.lower() in ['yellowfin','yf']:
    optimizer = YFOptimizer(
        curParams, lr=args.lr, mu=0.0, weight_decay=weight_decay, clip_thresh=2.0, curv_win_width=20)
else:
    raise Exception('Unsupported optimizer type encountered:'+args.optimizer)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

def train():
    #import cProfile, pstats
    #from io import StringIO
    #pr = cProfile.Profile()
    #pr.enable()


    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
        ssd_dim, means), AnnotationTransform())

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )

        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)

    for iteration in range(args.start_iter, max_iter):
        #t00 = time.time()
        doTimingStuff = False and iteration % 10 == 0
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )

            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        #images, targets, feats = next(batch_iterator)
        images, targets= next (batch_iterator)

        if args.cuda:
            images = images.cuda()
            images = Variable(images)
            if args.top_down_source != '':
                feats = feats.cuda()
                feats = Variable(feats.cuda())
            targets = [Variable(anno,volatile=True) for anno in targets]
            targets = [anno.cuda() for anno in targets]

            #targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        hints = None
        if args.use_hint:
            #hints = torch.zeros(len(targets), 20);  # was zeros! also try 1/20

            if args.top_down_source != '':
                #hints = hint_constants[1] * torch.ones (len (targets), 2048)
                #for irow in len(hints):
                hints = feats

            else:
                hints = hint_constants[1]*torch.ones(len(targets),20) # was zeros! also try 1/20
                #print('target type:',type(targets))
                #print('targets:',targets)
                for i_target,t in enumerate(targets): # if we're using a hint, we can't have more than one type of object per image.
                    target_class = t[:,-1]
                    assert len(set(target_class.data))==1,'cannot accept a heterogeneous hint (hint contains more than one class'
                    #print('target_class:',target_class)
                    hints[i_target,int(target_class.data[0])] = hint_constants[0]
                    #hints[i_target,target_class[0].data] = 1
                hints = Variable(hints)
            if args.cuda:
                hints = hints.cuda();

        t0 = time.time()
        # vgg, extra,loc,conf
        out = net(images,hints,args.hint_vgg,args.hint_extra,args.hint_loc,args.hint_conf)
        # backprop

        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()


        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())
        if args.visdom:
            viz.line(
                X=torch.ones((1, 3)).cpu() * iteration,
                Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                    loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )
            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                viz.line(
                    X=torch.zeros((1, 3)).cpu(),
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu(),
                    win=epoch_lot,
                    update=True
                )
        if iteration % 10 == 0:
            logger.log_value('loc loss', loss_l.data[0] , iteration)
            logger.log_value('conf loss', loss_c.data[0] , iteration)

            # log gradients.
            if False:
                for iprm,prm in enumerate(net.parameters()):
                    if prm.requires_grad:
                        #print(iprm)
                        logger.log_value('prm {}'.format(iprm), prm.grad.abs().mean().data[0], iteration)

        if (iteration+1) % args.checkpoint_freq == 0 or iteration == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), os.path.join(args.save_folder,'ssd300_0712_' +
                       repr(iteration+1) + '.pth'))

    torch.save(ssd_net.state_dict(), args.save_folder + '' + args.version + '.pth')




def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    if args.optimizer.lower() in ['yellowfin','yf']: # Note that this is different that the behaviour
        # for other kinds of optimizers since yellowfin adjusts its own learning rate,
        # this just multiplies it by a factor. This also relies on the fact that
        # adjust_learning_rate is only called during the steps.
        optimizer.set_lr_factor(optimizer.get_lr_factor() * gamma)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    train()
