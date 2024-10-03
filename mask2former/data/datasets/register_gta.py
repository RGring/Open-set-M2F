# Copyright (c) Facebook, Inc. and its affiliates.
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import PIL.Image
import numpy as np

CITYSCAPES_SEM_SEG_CATEGORIES = [
    {
        "color": [128, 64,128],
        "instances": True,
        "readable": "road",
        "name": "road",
        "evaluate": True,
    },
    {
        "color": [244, 35,232],
        "instances": True,
        "readable": "sidewalk",
        "name": "sidewalk",
        "evaluate": True,
    },
    {
        "color": [70, 70, 70],
        "instances": True,
        "readable": "building",
        "name": "building",
        "evaluate": True,
    },
    {
        "color": [102,102,156],
        "instances": True,
        "readable": "wall",
        "name": "wall",
        "evaluate": True,
    },
    {
        "color": [190,153,153],
        "instances": True,
        "readable": "fence",
        "name": "fence",
        "evaluate": True,
    },
    {
        "color": [153,153,153],
        "instances": True,
        "readable": "pole",
        "name": "pole",
        "evaluate": True,
    },
    {
        "color": [250,170, 30],
        "instances": True,
        "readable": "traffic light",
        "name": "traffic light",
        "evaluate": True,
    },
    {
        "color": [220,220,  0],
        "instances": True,
        "readable": "traffic sign",
        "name": "traffic sign",
        "evaluate": True,
    },
    {
        "color": [107,142, 35],
        "instances": True,
        "readable": "vegetation",
        "name": "vegetation",
        "evaluate": True,
    },
    {
        "color": [152,251,152],
        "instances": True,
        "readable": "terrain",
        "name": "terrain",
        "evaluate": True,
    },
    {
        "color": [70,130,180],
        "instances": True,
        "readable": "sky",
        "name": "sky",
        "evaluate": True,
    },
    {
        "color": [220, 20, 60],
        "instances": True,
        "readable": "person",
        "name": "person",
        "evaluate": True,
    },
    {
        "color": [255,  0,  0],
        "instances": True,
        "readable": "rider",
        "name": "rider",
        "evaluate": True,
    },
    {
        "color": [0,  0,142],
        "instances": True,
        "readable": "car",
        "name": "car",
        "evaluate": True,
    },
    {
        "color": [0,  0, 70],
        "instances": True,
        "readable": "truck",
        "name": "truck",
        "evaluate": True,
    },
    {
        "color": [255, 255, 255],
        "instances": True,
        "readable": "unknown",
        "name": "unknown",
        "evaluate": True,
    },
]

def _get_gta_meta():
    stuff_classes = [k["readable"] for k in CITYSCAPES_SEM_SEG_CATEGORIES if k["evaluate"]]
    stuff_colors = [k["color"] for k in CITYSCAPES_SEM_SEG_CATEGORIES if k["evaluate"]]
    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def load_gta(root, split, task):
    image_files = []
    image_file_list = f"{root}/{task}_{split}.txt"
    with open(image_file_list) as f:
        for line in f:
            image_files.append(f"{root}/images/{line.strip()}.png")
    image_files = list(sorted(image_files))

    examples = []
    for im_file in image_files:
        # Remove files that do not contain any objects, because they can not be handled by the loss function of Mask2Former
        if im_file.split('/')[-1] == "15617.png":
            continue

        # Get image size
        img = PIL.Image.open(im_file)
        seg = PIL.Image.open(im_file.replace('images', 'labels').replace('.png', f"_{task}.png"))
        w_seg, h_seg = seg.size
        w, h = img.size
    
        # Detectron2 transform can not handle image-segmentation pairs of different sizes
        if w != w_seg or h != h_seg:
            continue

        examples.append({
            "file_name": im_file,
            "sem_seg_file_name": im_file.replace('images', 'labels').replace('.png', f"_{task}.png"), 
            "height": h,
            "width": w,
        })
        
    return examples

def register_gta(root):
    root = os.path.join(root, "gta")
    meta = _get_gta_meta()
    
    for split in ["train", "val"]:
        for task in ["labelIds_15_w_openSet", "labelIds_15_wo_privates"]:
            name = f"gta_{task}_{split}"

            DatasetCatalog.register(
                name, lambda x=root, y=split, z=task: load_gta(x, y, z)
            )
            
            image_dir = os.path.join(root, "images")
            gt_dir = os.path.join(root, "labels")
            MetadataCatalog.get(name).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=255, 
                **meta,
            )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_gta(_root)