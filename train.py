import os
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
import numpy as np
import time

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
<<<<<<< HEAD
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
=======
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
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
<<<<<<< HEAD
parser.add_argument('--use_hint', default=False, type=str2bool, help='Use Network Priming')
parser.add_argument('--hint_vgg', default=False, type=str2bool, help='Apply priming to vgg stage')
parser.add_argument('--hint_extra', default=False, type=str2bool, help='Apply priming to extra layers stage')
parser.add_argument('--hint_loc', default=False, type=str2bool, help='Apply priming to localization layers stage')
parser.add_argument('--hint_conf', default=False, type=str2bool, help='Apply priming to classifications layers stage')

#args = parser.parse_args(['--voc_root','/home/amir/Dropbox/code/jacky','--cuda','1'])
args = parser.parse_args(['--voc_root','/home/amir/data/voc/VOCdevkit','--cuda','1','--visdom','False'])
#args = parser.parse_args()
=======
args = parser.parse_args()

>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = (v1, v2)[args.version == 'v2']

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

<<<<<<< HEAD
#train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
train_sets = [('2007_p', 'train')]
=======
train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
# train_sets = 'train'
ssd_dim = 300  # only support 300 now
means = (104, 117, 123)  # only support voc now
num_classes = len(VOC_CLASSES) + 1
batch_size = args.batch_size
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = 120000
weight_decay = 0.0005
stepvalues = (80000, 100000, 120000)
gamma = 0.1
momentum = 0.9

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_ssd('train', 300, num_classes)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

<<<<<<< HEAD
import itertools
if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
    if args.use_hint:
        print('Freezing all parameters except for hinting params.')
        for p in ssd_net.parameters():
            if type(p) is not nn.BatchNorm2d:
                p.requires_grad = False

        for p in itertools.chain(ssd_net.vgg_controllers.parameters(),
            ssd_net.extra_controllers.parameters(),
            ssd_net.loc_controllers.parameters(),
            ssd_net.conf_controllers.parameters()):
            p.requires_grad = True


else:
    assert not args.use_hint,'Not supporting learning to prime for a non-pretrained model'

=======
if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    net = net.cuda()


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

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

<<<<<<< HEAD
=======

>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
def train():
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
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
<<<<<<< HEAD

        hints = None

        if args.use_hint:
            hints = torch.zeros()
            for i_target,t in enumerate(targets): # if we're using a hint, we can't have more than one type of object per image.
                target_class = t[:,-1]
                assert len(target_class)==1,'cannot accept a heterogeneous hint (hint contains more than one class'
                hints[i_target.target_class[0]] = 1
            hints = Variable(hints,volatile=True)
            if args.cuda:
                hints = hints.cuda();

        t0 = time.time()
        # vgg, extra,loc,conf
        out = net(images,hints,args.hint_vgg,args.hint_extra,args.hint_loc,args.hint_conf)
=======
        t0 = time.time()
        out = net(images)
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
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
        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_0712_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), args.save_folder + '' + args.version + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
