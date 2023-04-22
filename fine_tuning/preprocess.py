from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from PIL import Image
import os


data_root = 'fine_tuning/TN-SCUI2020/segmentation/augtrain'

img_list = sorted(glob.glob(os.path.join(data_root,"image/*.bmp")))
msk_list = sorted(glob.glob(os.path.join(data_root,"mask/*.bmp")))

bbox_coords = {}
ground_truth_masks = {}
images = {}

for img_dir,msk_dir in zip(img_list,msk_list):
    
    index_key = os.path.basename(img_dir)
    
    gray_image = np.array(Image.open(img_dir))
    images[index_key] = gray_image
    
    mask = np.array(Image.open(msk_dir))
    ground_truth_masks[index_key] = (mask>0.5)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_SIMPLE)[-2:]

    x, y, w, h = cv2.boundingRect(contours[0]) #第一个为最大连通域，为目标区域
    height, width = gray_image.shape
    bbox_coords[index_key] = np.array([x, y, x + w, y + h])


# Helper functions provided in https://github.com/facebookresearch/segment-anything/blob/9e8f1309c94f1128a6e5c047a10fdcb02fc8d651/notebooks/predictor_example.ipynb
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
'''    
name = 'stampDS-00004'
image = cv2.imread(f'scans/scans/{name}.png')

plt.figure(figsize=(10,10))
plt.imshow(image)
show_box(bbox_coords[name], plt.gca())
show_mask(ground_truth_masks[name], plt.gca())
plt.axis('off')
plt.show()
plt.savefig(f"test.jpg")
'''
