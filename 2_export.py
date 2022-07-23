import os
import json
import utils.make_coco as make_coco


def load_json(coco_path):
    with open(coco_path) as file:
        coco = json.load(file)
    return coco

def save_json(coco_path, coco):
    with open(coco_path, 'w') as outfile:
        json.dump(coco, outfile)

def create_coco():
    return {
        "info": make_coco.add_info(),
        "licenses": make_coco.add_licenses(),
        "categories": [
            {
                "supercategory": 'c',
                "id": 0,
                "name": 'noncon',
            },
            {
                "supercategory": 'c',
                "id": 1,
                "name": 'h_rect',
            },
            {
                "supercategory": 'c',
                "id": 2,
                "name": 'square',
            },
            {
                "supercategory": 'c',
                "id": 3,
                "name": 'v_rect',
            },
        ],
        "images": [],
        "annotations": [],
    }

if __name__ == "__main__":

    path = 'out'
    files = [f for f in os.listdir(path) if f.endswith('.json')]

    images = []
    annotations = []
    count_image = 0
    count_annotation = 0

    for file in files:
        label = load_json('out/' + file)

        for i in range(3):
            spine = label['spine'][i]
            coco_object = {
                'category_id': spine['shape'],
                'image_id': count_image,
                'id': count_annotation,
                'iscrowd': 0,
                'area': spine['area'],
                'bbox': spine['bbox'],
                'segmentation': spine['segmentation'],
            }
            count_annotation += 1
            annotations.append(coco_object)

        coco_image = {
            "id": count_image,
            "height": label["height"],
            "width": label["width"],
            "file_name": label["filename"],
            "license": 1,
        }
        count_image += 1
        images.append(coco_image)

    cocos = create_coco()
    cocos['images'] = images
    cocos['annotations'] = annotations

    coco_path = 'out.json'
    save_json(coco_path, cocos)


