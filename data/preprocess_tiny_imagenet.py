import os
import shutil
import pandas as pd

def main():
    BASE = './tiny-imagenet-200'
    tr_target = os.path.join(BASE, 'train')
    val_target = os.path.join(BASE, 'val')
    if not os.path.isdir(os.path.join(BASE, 'train_preprocess')):
        os.mkdir(os.path.join(BASE, 'train_preprocess'))
    if not os.path.isdir(os.path.join(BASE, 'valid_preprocess')):
        os.mkdir(os.path.join(BASE, 'valid_preprocess'))
    tr_destination = os.path.join(BASE, 'train_preprocess')
    val_destination = os.path.join(BASE, 'valid_preprocess')

    tr_target_list = os.listdir(tr_target)
	
    for dir_ in tr_target_list:
        img_list = os.listdir(os.path.join(tr_target, dir_, 'images'))
        if not os.path.isdir(os.path.join(tr_destination, dir_)):
            os.mkdir(os.path.join(tr_destination, dir_))
        if not os.path.isdir(os.path.join(val_destination, dir_)):
            os.mkdir(os.path.join(val_destination, dir_))
        for img in img_list:
            img = os.path.join(tr_target, dir_, 'images', img)
            dest = os.path.join(tr_destination, dir_)
            if not os.path.isdir(os.path.join(dest, dir_, img)):
                shutil.copy(img, dest)
	
    with open(os.path.join(val_target, "val_annotations.txt"), 'r') as val_info:
        lines = val_info.readlines()

    val_info = pd.read_csv(os.path.join(val_target, "val_annotations.txt"),
                           sep = "\t",
                           names=['img', 'label', 'bx1', 'bx2','bx3', 'bx4'],
                           header=None)

    val_img = list(val_info['img'])
    val_label = list(val_info['label'])
    assert len(val_img) == len(val_label)

    for img, label in zip(val_img, val_label):
        img = os.path.join(val_target, 'images', img)
        dest = os.path.join(val_destination, label)
        shutil.copy(img, dest)

if __name__ == "__main__":
    main()