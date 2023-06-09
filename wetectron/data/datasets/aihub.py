import os
import pickle

import torch
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET

from wetectron.structures.bounding_box import BoxList

class AIHubDataset(torch.utils.data.Dataset):

    CLASSES = (
        "wheelchair",
        "truck",
        "tree_trunk",
        "traffic_sign",
        "traffic_light",
        "traffic_light_controller",
        "table",
        "stroller",
        "stop",
        "scooter",
        "potted_plant",
        "pole",
        "person",
        "parking_meter",
        "power_controller",
        "movable_signage",
        "motorcycle",
        "kiosk",
        "fire_hydrant",
        "dog",
        "chair",
        "cat",
        "carrier",
        "car",
        "bus",
        "bollard",
        "bicycle",
        "bench",
        "barricade",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, proposal_file=None):
        self.root = data_dir
        self.set = split
        self.keep_occluded = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "images", "MP_SEL_%s.jpg")
        self._setpath = os.path.join(self.root, "set", "%s.txt")
        
        with open(self._setpath % self.set) as f:
            self.setinfo = f.readlines()
        self.setinfo = [x.strip("\n") for x in self.setinfo]

        self.img_ids = []
        self.anno_ids = []
        for x in self.setinfo :
            x = x.split("\t")
            self.img_ids.append(x[0])
            self.anno_ids.append([x[1], int(x[2])])

        cls = AIHubDataset.CLASSES
        self.class_to_idx = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))
        self.proposals = None

    def get_origin_id(self, index):
        id = self.img_ids[index]
        return id

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        anno_id = self.anno_ids[index][0]
        anno_idx = self.anno_ids[index][1]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        if not os.path.exists(self._annopath % anno_id):
            target = None
        else:
            target = self.get_groundtruth(index)
            target = target.clip_to_image(remove_empty=True)
                
        rois = None

        if self.transforms is not None:
            img, target, rois = self.transforms(img, target, rois)

        return img, target, rois, index

    def __len__(self):
        return len(self.img_ids)

    def get_groundtruth(self, index):
        anno_id = self.anno_ids[index][0]
        anno_idx = self.anno_ids[index][1]
        anno_id, anno_idx = self.anno_ids[index]
        image = ET.parse(self._annopath % anno_id).findall("image")[anno_idx]
        anno = self._preprocess_annotation(image)
        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("occluded", anno["occluded"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        classes = []
        occluded_boxes = []
        TO_REMOVE = 1

        for box in target.iter("box"):
            occluded = (int(box.attrib['occluded']) == 1)
            # if not self.keep_occluded and occluded:
            #     continue
            cls = box.attrib['label']
            bbox = [
                float(box.attrib['xtl']), 
                float(box.attrib['ytl']), 
                float(box.attrib['xbr']), 
                float(box.attrib['ybr']),
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, bbox)))
            )
            boxes.append(bndbox)
            classes.append(self.class_to_idx[cls])
            occluded_boxes.append(occluded)

        im_info = tuple(map(int, (target.attrib['height'], target.attrib['width'])))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(classes),
            "occluded": torch.tensor(occluded_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.img_ids[index]
        anno_id = self.anno_ids[index][0]
        anno_idx = self.anno_ids[index][1]
        anno_id, anno_idx = self.anno_ids[index]
        file_name = self._imgpath % img_id
        if os.path.exists(self._annopath % anno_id):
            image = ET.parse(self._annopath % anno_id).findall("image")[anno_idx]
            im_info = tuple(map(int, (image.attrib['height'], image.attrib['width'])))
            return {"height": im_info[0], "width": im_info[1], "file_name": image.attrib['name']}
        else:
            name = os.path.join(file_name)
            img = Image.open(name).convert("RGB")
            return  {"height": img.size[1], "width": img.size[0], "file_name": "%s.jpg"%index}
        
    def map_class_id_to_class_name(self, class_id):
        return AIHubDataset.CLASSES[class_id]
