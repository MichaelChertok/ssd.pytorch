<<<<<<< HEAD
import os

=======
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
<<<<<<< HEAD

from data import v2
from layers import *
# Auxiliary function to allow the gradual training of the network - this actually allows the last <howMuch> layers to be learned
# (if they could be learned before).
def disable_layers(L,howMuch):
    L = list(reversed(L))
    indices = [i for i, x in enumerate(L) if x == True]
    for i in range(0,max(0,len(indices)-howMuch)):
        L[indices[i]]=False

    return list(reversed(L))
=======
from layers import *
from data import v2
import os

>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """
<<<<<<< HEAD
    def makeControllerLayer(self,m):
        #print(self.controllerBias)
        M = 2 if self.controller_bias else 1
        L = [nn.Linear(self.hintSize, M * m.out_channels, bias=False)]
        return nn.Sequential(*L)

    def __init__(self, phase, base, extras, head, num_classes, controller_bias=False, controller_sigmoid=True, residual_controllers=False,add_relu=False,
                 n_to_learn = [100,100,100,100], add_hint_interpreter=False):
=======

    def __init__(self, phase, base, extras, head, num_classes):
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 300
<<<<<<< HEAD
        self.residual_controllers = residual_controllers
        self.add_relu = add_relu
=======

>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
<<<<<<< HEAD
        hintSize = 20
        self.hintSize=hintSize
        if add_hint_interpreter:
            self.hint_interpreter = nn.Sequential(nn.Linear(2048,hintSize,bias=True))#,nn.ReLU())
        self.add_hint_interpreter = add_hint_interpreter
        vgg_controllers = []
        vgg_is_controller = []
        extra_controllers = []
        extra_is_controller = []
        loc_is_controller = []
        loc_controllers = []
        conf_controllers = []
        conf_is_controller = []
        self.controller_sigmoid = controller_sigmoid
        self.controller_bias = controller_bias
        #print('-----------------------',controllerBias)
         # add controllers.
        
        for m in self.vgg:
            if type(m) is nn.Conv2d:
                vgg_controllers.append(self.makeControllerLayer(m))
                vgg_is_controller.append(True)
                #print('--------------------')
                #print(vgg_controllers[-1])
            else:
                vgg_controllers.append(nn.Sequential())
                vgg_is_controller.append(False)
        for m in self.extras:
            if type(m) is nn.Conv2d:
                extra_controllers.append(self.makeControllerLayer(m))
                extra_is_controller.append(True)
            else:
                extra_controllers.append(nn.Sequential())
                extra_is_controller.append(False)
        for m in self.loc:
            if type(m) is nn.Conv2d:
                loc_controllers.append(self.makeControllerLayer(m))
                loc_is_controller.append(True)
            else:
                loc_controllers.append(nn.Sequential())
                loc_is_controller.append(False)
        for m in self.conf:
            if type(m) is nn.Conv2d:
                conf_controllers.append(self.makeControllerLayer(m))
                conf_is_controller.append(True)
            else:
                conf_controllers.append(nn.Sequential())
                conf_is_controller.append(False)

        self.vgg_controllers = nn.ModuleList(vgg_controllers)
        self.extra_controllers = nn.ModuleList(extra_controllers)
        self.loc_controllers = nn.ModuleList(loc_controllers)
        self.conf_controllers = nn.ModuleList(conf_controllers)
        
        self.vgg_is_controller=disable_layers(vgg_is_controller,n_to_learn[0])
        self.extra_is_controller=disable_layers(extra_is_controller,n_to_learn[1])
        self.loc_is_controller=disable_layers(loc_is_controller,n_to_learn[2])
        self.conf_is_controller=disable_layers(conf_is_controller,n_to_learn[3])
        #print('-----------------')
        #print(n_to_learn)
        #print(zip(self.vgg_is_controller,self.
        #for k in self.vgg_is_controller:
        #    assert k==False
=======
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
<<<<<<< HEAD
    


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
    def applyController(self,x,hint,controller,ovr,isController):
        if not isController:
            return x
    
        if not ovr or controller is None or hint is None or ovr == False:
            return x
        if type(controller) is nn.Sequential and len(controller) == 0:  # this is an empty controller, for prettier code...
            return x
        S = x.size()
        # print('size of x',S)
        # print('controller:',controller)

        h = controller(hint)

        if self.controller_bias:
            s = int(h.size()[1])//2
            h_mult = h[:,:s].contiguous()
            h_bias = h[:,s:].contiguous()
            h_mult = h_mult.view(S[0], S[1], 1, 1)
            h_bias = h_bias.view(S[0], S[1], 1, 1)
            if self.controller_sigmoid:
                h_mult = F.tanh(h_mult)
            x_result = x * h_mult + h_bias
        else:
            h = h.view(S[0], S[1], 1, 1).contiguous()
            if self.controller_sigmoid:
                h = F.tanh(h)
            x_result = x * h

        if self.add_relu:
            x_result = F.relu(x_result)
        if self.residual_controllers:
            x = x_result + x
            #return x + x_res * h
        else:
            x = x_result
            #return x_res

        #print('size of controlled:',x.size())
        return x

    def expand_ovr(self,ovr,N):
        if ovr is None:
            ovr = [True]*N
        elif type(ovr) is bool:
            ovr = [ovr]*N
        return ovr

    def forward(self, x, hint = None, hint_override_vgg = None, extra_ovr = None,
                loc_ovr = None, conf_ovr = None):
=======

    def forward(self, x):
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

<<<<<<< HEAD
        # check which layers need to be hinted.

        hint_override_vgg = self.expand_ovr(hint_override_vgg,len(self.vgg))
        loc_ovr = self.expand_ovr(loc_ovr, len(self.loc))
        conf_ovr = self.expand_ovr(conf_ovr, len(self.conf))
        extra_ovr = self.expand_ovr(extra_ovr, len(self.extras))

        run_up_to = 23
        vgg_length = len(self.vgg)
        #assert vgg_length in [35, 46], 'unexpected vgg length'
        if len(self.vgg) > 35:  # this means no batch-norm.
            run_up_to = 33  # batch norm, other layer...

        # apply vgg up to conv4_3 relu
        
        if self.add_hint_interpreter and hint is not None:
            hint = self.hint_interpreter(hint)
        
        for k in range(run_up_to):
            #print ('---------------{}----------------'.format(k))
            x = self.vgg[k](x)
            x = self.applyController(x, hint, self.vgg_controllers[k], hint_override_vgg[k],self.vgg_is_controller[k])
=======
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
<<<<<<< HEAD

        for k in range(run_up_to, len(self.vgg)):
            x = self.vgg[k](x)
            x = self.applyController(x, hint, self.vgg_controllers[k], hint_override_vgg[k],self.vgg_is_controller[k])
=======
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
<<<<<<< HEAD
            x = v(x)
            x = self.applyController(x,hint,self.extra_controllers[k],extra_ovr[k],self.extra_is_controller[k])
            x = F.relu(x)#, inplace=True)
=======
            x = F.relu(v(x), inplace=True)
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
<<<<<<< HEAD
        for (x, l, c, lc, cc, l_ovr, c_ovr, is_l,is_c) in zip(sources, self.loc, self.conf, self.loc_controllers, self.conf_controllers,
                                     loc_ovr,conf_ovr, self.loc_is_controller, self.conf_is_controller):
            L = l(x)
            C = c(x)
            L = self.applyController(L, hint, lc,l_ovr, is_l)
            C = self.applyController(C, hint, cc, c_ovr, is_c)

            loc.append(L.permute(0, 2, 3, 1).contiguous())
            conf.append(C.permute(0, 2, 3, 1).contiguous())
=======
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


<<<<<<< HEAD
def multibox(vgg, extra_layers, cfg, num_classes, used_bn=False):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2] # these numbers rely on the fact that there are no batch norm layers.
    if used_bn:
        vgg_source = [34,46]

=======
def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
<<<<<<< HEAD
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
=======
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


<<<<<<< HEAD
def build_ssd(phase, size=300, num_classes=21, controller_bias = False, controller_sigmoid=False, insertBNLayers=False,
              residual_controllers=False, add_relu=False, n_to_learn=[100,100,100,100],add_hint_interpreter=False):
=======
def build_ssd(phase, size=300, num_classes=21):
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return
<<<<<<< HEAD
    return SSD(phase, *multibox(vgg(base[str(size)], 3, batch_norm=insertBNLayers),
                                add_extras(extras[str(size)], 1024),
                                mbox[str(size)], num_classes, insertBNLayers),
               num_classes,
               controller_bias=controller_bias,
               controller_sigmoid=controller_sigmoid,
               residual_controllers=residual_controllers,
               add_relu=add_relu,n_to_learn=n_to_learn,
               add_hint_interpreter=add_hint_interpreter)
=======

    return SSD(phase, *multibox(vgg(base[str(size)], 3),
                                add_extras(extras[str(size)], 1024),
                                mbox[str(size)], num_classes), num_classes)
>>>>>>> 197f922fbb11c9d7e2b6c75d9467e337eb18138a
