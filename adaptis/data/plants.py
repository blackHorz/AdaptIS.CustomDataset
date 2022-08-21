import cv2
import numpy as np
from pathlib import Path
from .base import BaseDataset



class PlantsDataset(BaseDataset):
    def __init__(self, dataset_path, split='train', use_jpeg=False, **kwargs):
        super(PlantsDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split

        images_path = self.dataset_path / split

        images_mask = '*rgb.png'
        images_list = sorted(images_path.rglob(images_mask))

        self.dataset_samples = []
        for image_path in images_list:
            image_name = str(image_path.relative_to(images_path))
            instances_name = image_name.replace(images_mask[1:], 'im.png')
            instances_path = str(images_path / instances_name)

            semantic_name = image_name.replace(images_mask[1:], 'im.png')
            semantic_path = str(images_path / semantic_name)

            self.dataset_samples.append((str(image_path), instances_path, semantic_path))

        total_classes = 6
        self._resize =  (768, 768)

    def get_sample(self, index):
        image_path, instances_path, semantic_path = self.dataset_samples[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = cv2.imread(instances_path, cv2.IMREAD_GRAYSCALE)
        instance_map, label_map = self.Convert2ClassIns(img)
        bResize =  True
        if bResize:
            image = cv2.resize(image, (self._resize), interpolation = cv2.INTER_AREA)
            instance_map = cv2.resize(instance_map.astype(np.float32), (self._resize), interpolation = cv2.INTER_AREA).astype(np.int32)
            label_map = cv2.resize(label_map.astype(np.float32), (self._resize), interpolation = cv2.INTER_AREA).astype(np.int32)
        
        instances_info = dict()
        instances_ids = self.get_unique_labels(instance_map)
        for obj_id in instances_ids:
            class_id = self.getClassId(obj_id) 
            ignore = False
            instances_info[obj_id] = {
                'class_id': class_id, 'ignore': ignore
            }

        sample = {
            'image': image,
            'instances_mask': instance_map,
            'instances_info': instances_info,
            'semantic_segmentation': label_map
        }
        # print(type(image), image.shape)
        # print(type(instance_map), instance_map.shape)
        # print(type(instances_info), instances_info)
        # print(type(label_map), label_map.shape)
        return sample
    def getClassId(self, objId):
        assert objId >= 0, 'ObjectId not known'
        objStr = str(objId)
        if len(objStr) == 1:
            return 0
        else:
            return int(objStr[0])

    def Convert2ClassIns(self, img):
        img = img / 4
        w, h = img.shape
        Instance_ = np.zeros((w,h), np.int32)
        Class_ = np.zeros((w,h), np.int32)
        for i in range(w):
            for j in range(h):
                if img[i,j] != 0:
                    tmp = str(img[i, j])
                    Class_[i,j] = int(tmp[0])
                    Instance_[i,j] = img[i, j]
        return Instance_, Class_

    @property
    def stuff_labels(self):
        return [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]

    @property
    def things_labels(self):
        return [24, 25, 26, 27, 28, 31, 32, 33]
