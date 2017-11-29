"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('--use_hint', default=False, type=str2bool, help='Use Network Priming')
parser.add_argument('--hint_vgg', default=False, type=str2bool, help='Apply priming to vgg stage')
parser.add_argument('--hint_extra', default=False, type=str2bool, help='Apply priming to extra layers stage')
parser.add_argument('--hint_loc', default=False, type=str2bool, help='Apply priming to localization layers stage')
parser.add_argument('--hint_conf', default=False, type=str2bool, help='Apply priming to classifications layers stage')
parser.add_argument('--use_post_hint', default=False, type=str2bool, help='Hint the network as postprocessing stage.')
parser.add_argument('--controller_bias', default=False, type=str2bool, help='Learn bias factor in controllers')
parser.add_argument('--controller_sigmoid', default=False, type=str2bool, help='Use sigmoid before applying controllers')
parser.add_argument('--residual_controllers',default=False,type=str2bool,help='use residual controllers')
parser.add_argument('--insert_bn_layers',default=False,type=str2bool,help='resinert BN layers before training (set this to true if you did during training')
parser.add_argument('--add_noise', default=0, type=float, help='Add noise to test images')
parser.add_argument('--validation_frac', default=1, type=int, help='1 / fraction of validation set used to evaluate. ')
parser.add_argument('--add_relu',default=False,type=str2bool,help='apply ReLu after priming layer')
parser.add_argument('--test_set', default='val', help='on what to test? (val/test)')
parser.add_argument('--sanity_value',type=float,default=1,help='just for debugging the hints')
parser.add_argument('--hint_interpreter',default=False,type=str2bool,help='add extra layer after hint')
parser.add_argument('--hint_constants',default='1_0',help='set value of positive/negative hinted classes to pos_neg')
parser.add_argument('--quit_if_exists',default='_',help='quit process if given file exists.')
parser.add_argument('--all_priming',default=False,type=str2bool,help='prime for all classes')
parser.add_argument('--external_gt',default='',help='allow external file to set priming for each of the image ids')
parser.add_argument('--detections_file_name',default='',help='output file for all detections')
parser.add_argument('--clip_noise',default=False,help='clip noisy input images to -128,128')
parser.add_argument('--allow-multiclass-hint',type=str2bool, default=False,help='allow hint of more than one class?')
parser.add_argument('--force-single-class',default=-1,type=int,help='force the "ground truth" class to be a specific value , set to -1 to avoid (default)')
parser.add_argument('--contextual-priming',default=False,type=str2bool,help='use inferred context hints')
parser.add_argument('--top-down-source',default='',help='filename for datasource of top-down image data (e.g, resnet results')
parser.add_argument('--n_to_learn',default='100_100_100_100',help='how many priming layer to actually learn in each network segment')



#parser.add_argument('--uid',default='',help='string to add to results file template to avoid conflicting result files')

from uuid import uuid4
args = parser.parse_args()
if args.quit_if_exists != '_' and os.path.isfile((args.quit_if_exists)):
    print('file already exists:',args.quit_if_exists,' - quitting')
    sys.exit()

n_to_learn = [int(q) for q in args.n_to_learn.split('_')]
if args.top_down_source!='':
    assert args.use_post_hint==False


if args.external_gt != '':
    external_gt = [int(k.strip()) for k in open(args.external_gt).readlines()]
else:
    external_gt = None;

hint_constants = [float(q) for q in args.hint_constants.split('_')]


id_to_pred = None
if args.contextual_priming:
    id_to_pred = torch.load('/home/amir/resnet101_voc/id_to_prediction.pth')


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = os.path.join(args.voc_root, 'VOC' + YEAR)
dataset_mean = (104, 117, 123)
#assert args.test_set in ['val','test'],'test set must be either val or test...'
set_type = args.test_set
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir

def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s_ph%s_%s.txt' % (cls, args.use_post_hint,my_uuid)
    #filedir = os.path.join(devkit_path, 'results')
    filedir = os.path.join (args.save_folder, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path

def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default False)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile) or True: # ignore pesky caching
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            #if i % 100 == 0:
            #    print('Reading annotation for {:d}/{:d}'.format(
            #        i + 1, len(imagenames)))
        # save
        #print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

####################
def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    dataset.isTraining=False
    if args.allow_multiclass_hint==True and args.use_hint and not args.contextual_priming and not args.top_down_source!='':
        # call our method on union of all hints.
        test_net_hint_union(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05)
        return
    
    num_images = len(dataset)
    

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir(save_folder, set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        #print('current id:',dataset.ids[i])
        #im, gt, h, w, feats = dataset.pull_item(i)
        im, gt, h, w = dataset.pull_item (i)
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        if args.add_noise > 0:
            x.data+=(torch.randn(x.size())*args.add_noise)
            if args.clip_noise:
                x.data[x.data<-128]=-128
                x.data[x.data>=128]=128
        _t['im_detect'].tic()
        if args.contextual_priming:
            zz = (id_to_pred[dataset.ids[i][-1]] > .1).nonzero()

            #zz = id_to_pred[dataset.ids[i][-1]]
            #v,order=id_to_pred[dataset.ids[i][-1]].sort()
            #order=set (order.view (order.numel ())[-3:])
            #ground_truth_classes=order
            ground_truth_classes = set(list(zz.view(zz.numel())))
        else:
            ground_truth_classes = set([int(q) for q in gt[:,-1]])
        #print('no. ground truth classes:',len(ground_truth_classes))
        if args.force_single_class!=-1:
            ground_truth_classes = [args.force_single_class]
            #print('forcing single class:',ground_truth_classes)
        if args.top_down_source == '' or not args.use_hint:
            assert not args.use_hint or args.allow_multiclass_hint==True or len(ground_truth_classes) <=1,'I do not support heterogeneous images - set --allow-multiclass-hint to do so'
        #print
        detections = None
        # use multiple hints...

        hints=None


        if args.use_hint:
            if args.top_down_source != '':

                feats = feats.cuda ()
                feats = Variable (feats.cuda ())
                # hints = hint_constants[1] * torch.ones (len (targets), 2048)
                # for irow in len(hints):
                hints = feats
                hints = hints.cuda ();
            else:
                hints = hint_constants[1] * torch.ones(1,20)
                if external_gt is not None:
                    hints[0,int( external_gt[i])]= hint_constants[0]
                else:
                    for ggg in ground_truth_classes:
                        hints[0,int( ggg)] = hint_constants[0]
                    #hints[0,int( gt[0,-1])]= hint_constants[0]
                hints = Variable(hints)*args.sanity_value

                hints = hints.cuda();



        detections = net(x,hints,args.hint_vgg,args.hint_extra,args.hint_loc,args.hint_conf).data

        if args.use_post_hint:
            if external_gt is None:
                assert args.allow_multiclass_hint==True or len(ground_truth_classes) <=1,'I do not support heterogeneous images - set --allow-multiclass-hint to do so'
                
                #assert len(set(gt[:,-1]))==1, 'I do not support heterogeneous images for hinting for now.'
                #ground_truth_class = int(gt[0,-1])
            else:
                ground_truth_class = external_gt[i]

        #    detections[:,0,:,:] = -1.0
        #    for i_class in range(21):
        #        # zero out all detections except those of the ground-truth
        #        if i_class != ground_truth_class+1:
        #           detections[:,i_class,:,:] = -10.0

        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):

            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()

            if args.use_post_hint: # remove all non-hint classes.
                if j-1 not in ground_truth_classes:
                    scores[:]=-10
            #    else:
                    #pass
                    #scores[:]=-1000
            #        scores[:] = -1
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            #if args.use_post_hint and ground_truth_class != j-1:
            #    continue
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)
    
####################


def test_net_hint_union(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    assert args.force_single_class==-1 and args.use_hint
    num_images = len(dataset)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir(save_folder, set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        #print('current id:',dataset.ids[i])
        im, gt, h, w = dataset.pull_item(i)
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        if args.add_noise > 0:
            x.data+=(torch.randn(x.size())*args.add_noise)
            if args.clip_noise:
                x.data[x.data<-128]=-128
                x.data[x.data>=128]=128
        _t['im_detect'].tic()

        ground_truth_classes = set([int(q) for q in gt[:,-1]])

        assert args.allow_multiclass_hint or len(ground_truth_classes) <=1,'I do not support heterogeneous images - set --allow-multiclass-hint to do so'
        
        ground_truth_classes = list(ground_truth_classes)
        #print('no. ground truth classes:',len(ground_truth_classes))
        hints = hint_constants[1] * torch.ones(1,20)
        hints[0,int( ground_truth_classes[0])]= hint_constants[0]
        hints = Variable(hints)*args.sanity_value
        hints = hints.cuda();
        # use multiple hints...
        detections = net(x,hints,args.hint_vgg,args.hint_extra,args.hint_loc,args.hint_conf).data
        if False:
            for i_g in range(1,len(ground_truth_classes)):
                myHint = ground_truth_classes[i_g]
                hints = hint_constants[1] * torch.ones(1,20)
                hints[0,int( myHint)]= hint_constants[0]
                hints = Variable(hints)*args.sanity_value
                hints = hints.cuda();
                detections2 = net(x,hints,args.hint_vgg,args.hint_extra,args.hint_loc,args.hint_conf).data
                detections[:,myHint+1,:,:] = detections2[:,myHint+1,:,:]

        if args.use_post_hint:
           
            if external_gt is None:
                assert args.allow_multiclass_hint or len(ground_truth_classes) <=1,'I do not support heterogeneous images - set --allow-multiclass-hint to do so'
                
                #assert len(set(gt[:,-1]))==1, 'I do not support heterogeneous images for hinting for now.'
                #ground_truth_class = int(gt[0,-1])
            else:
                ground_truth_class = external_gt[i]

        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()

            if args.use_post_hint: # remove all non-hint classes.
                if j-1 not in ground_truth_classes:
                    scores[:]=-10
            #    else:
                    #pass
                    #scores[:]=-1000
            #        scores[:] = -1
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            #if args.use_post_hint and ground_truth_class != j-1:
            #    continue
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)
    
    
def test_net_all_priming(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    
    assert args.force_single_class==-1
    dataset.isTraining=False
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir(save_folder, set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        #print('current id:',dataset.ids[i])
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))

        if args.cuda:
            x = x.cuda()
        if args.add_noise > 0:
            x.data+=(torch.randn(x.size())*args.add_noise)
        _t['im_detect'].tic()
        
        myHint = 0
        if args.use_hint:
            hints = hint_constants[1] * torch.ones(1,20)
            hints[0,int( myHint)]= hint_constants[0]
            hints = Variable(hints)*args.sanity_value
            hints = hints.cuda();
        else:
            raise Exception('False value for args.use_hint not suppored with all_priming')
            
        detections = net(x,hints,args.hint_vgg,args.hint_extra,args.hint_loc,args.hint_conf).data
        
        for myHint in range(1,20):
            hints = hint_constants[1] * torch.ones(1,20)
            hints[0,int( myHint)]= hint_constants[0]
            hints = Variable(hints)*args.sanity_value
            hints = hints.cuda();
            detections2 = net(x,hints,args.hint_vgg,args.hint_extra,args.hint_loc,args.hint_conf).data
            detections[:,myHint+1,:,:] = detections2[:,myHint+1,:,:]
        
        # We're assuming that the
        #if args.use_post_hint:
        #    assert len(set(gt[:,-1]))==1, 'I do not support heterogeneous images for hinting for now.'
        #    ground_truth_class = int(gt[0,-1])

        #    detections[:,0,:,:] = -1.0 
        #    for i_class in range(21):
        #        # zero out all detections except those of the ground-truth
        #        if i_class != ground_truth_class+1:
        #           detections[:,i_class,:,:] = -10.0

        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):

            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()

            #if args.use_post_hint: # remove all non-hint classes.
            #    if j-1 != ground_truth_class:
            #        scores[:]=-10
            #    else:
                    #pass
                    #scores[:]=-1000
            #        scores[:] = -1
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            #if args.use_post_hint and ground_truth_class != j-1:
            #    continue
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


if __name__ == '__main__':

    my_uuid = str(uuid4())

    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    if args.top_down_source != '':
        args.hint_interpreter = True
    net = build_ssd('test', 300, num_classes, controller_bias=args.controller_bias, controller_sigmoid=args.controller_sigmoid,  # initialize SSD
                        insertBNLayers=args.insert_bn_layers, residual_controllers=args.residual_controllers,add_relu=args.add_relu,
                    add_hint_interpreter = args.hint_interpreter,n_to_learn=n_to_learn)


    # allow loading of original model easily.
    oldNetDict = torch.load (args.trained_model)
    new_net_dict = net.state_dict ()
    for k in oldNetDict.keys ():
        if k in new_net_dict:
            new_net_dict[k] = oldNetDict[k]
    net.load_state_dict (new_net_dict)

    #net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = VOCDetection(args.voc_root, [('2007', set_type)], BaseTransform(300, dataset_mean), AnnotationTransform(),
                           dataset_frac=args.validation_frac)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    if args.all_priming:
        test_net_all_priming(args.save_folder, net, args.cuda, dataset,
                BaseTransform(net.size, dataset_mean), args.top_k, 300,
                thresh=args.confidence_threshold)
    else:
        test_net(args.save_folder, net, args.cuda, dataset,
                BaseTransform(net.size, dataset_mean), args.top_k, 300,
                thresh=args.confidence_threshold)
