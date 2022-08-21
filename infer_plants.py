
#import ipywidgets as widgets
import matplotlib.pyplot as plt

import cv2
import sys
import os
import torch
from tqdm import tnrange
import pylab

from adaptis.inference.adaptis_sampling import get_panoptic_segmentation
from adaptis.inference.prediction_model import AdaptISPrediction
from adaptis.data.plants import PlantsDataset
from adaptis.model.cityscapes.models import get_cityscapes_model

device = torch.device('cuda')

dataset_path = '/home/fftai/working/pytorch/adaptis.pytorch-master/custom_dataset/custom_dataset/'
weights_path = '/home/fftai/working/pytorch/adaptis.pytorch-master/experiments/plants/000/checkpoints/proposals_last_checkpoint.pth'
dataset = PlantsDataset(dataset_path, split='val', with_segmentation=True)


model = get_cityscapes_model(num_classes=6, norm_layer=torch.nn.BatchNorm2d, backbone='resnet50', with_proposals=True)
pmodel = AdaptISPrediction(model, dataset, device)

pmodel.net.load_state_dict(torch.load(weights_path)['model_state'])
proposals_sampling_params = {
    'thresh1': 0.4,
    'thresh2': 0.5,
    'ithresh': 0.3,
    'fl_prob': 0.10,
    'fl_eps': 0.003,
    'fl_blur': 2,
    'max_iters': 100
}

image_path = '/home/fftai/working/pytorch/adaptis.pytorch-master/custom_dataset/custom_dataset/val/stn1_syn006_pkg000_0_1_rep_rgb.png'

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pred = get_panoptic_segmentation(pmodel, image,
                                 sampling_algorithm='proposals',
                                 use_flip=True, **proposals_sampling_params)


pylab.imshow(pred)
def show(ix):
    import pylab
    pylab.figure(figsize=(20,10))
    pylab.imshow((pred['instances_mask'] == ix).astype('float32')[...,None] * 0.5  + image.astype('float32')/255/2)

#widgets.interact(show, ix=widgets.BoundedIntText(min=0, max=len(pred['masks']), value=0))




from adaptis.coco.panoptic_metric import PQStat, pq_compute, print_pq_stat

def test_model(pmodel, dataset,
               sampling_algorithm, sampling_params,
               use_flip=False, cut_radius=-1):
    pq_stat = PQStat()
    categories = dataset._generate_coco_categories()
    categories = {x['id']: x for x in categories}

    for indx in tnrange(len(dataset)):
        sample = dataset.get_sample(indx)
        pred = get_panoptic_segmentation(pmodel, sample['image'],
                                         sampling_algorithm=sampling_algorithm,
                                         use_flip=use_flip, cut_radius=cut_radius, **sampling_params)
        
        
        coco_sample = dataset.convert_to_coco_format(sample)
        pred = dataset.convert_to_coco_format(pred)

        pq_stat = pq_compute(pq_stat, pred, coco_sample, categories)
    
    print_pq_stat(pq_stat, categories)



test_model(pmodel, dataset,
           sampling_algorithm='proposals',
           sampling_params=proposals_sampling_params,
           use_flip=True)


