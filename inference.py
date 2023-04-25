import cv2
from sklearn.mixture import GaussianMixture


import argparse
import os
import random
import numpy as np
import torch
import io
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from  PIL  import  Image
from lang_sam import LangSAM
transform  = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.RandomGrayscale(p=0.05),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.3),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])





model = LangSAM()
def compute_heatmap(points, image_size, k_ratio=3.0):
    points = np.asarray(points)
    heatmap = np.zeros((image_size[0], image_size[1]), dtype=np.float32)
    n_points = points.shape[0]
    for i in range(n_points):
        x = points[i, 0]
        y = points[i, 1]
        col = int(x)
        row = int(y)
        try:
            heatmap[col, row] += 1.0
        except:
            col = min(max(col, 0), image_size[0] - 1)
            row = min(max(row, 0), image_size[1] - 1)
            heatmap[col, row] += 1.0
    k_size = int(np.sqrt(image_size[0] * image_size[1]) / k_ratio)
    if k_size % 2 == 0:
        k_size += 1
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = heatmap.transpose()
    return heatmap

def run_inference(net, image_pil): 
    objects = ['cup', 'drawer', 'potlid', 'microwave']
    bboxes = []
    for obj in objects: 
        with torch.no_grad(): 
            masks, boxes, phrases, logits = model.predict(image_pil, obj)
        bboxes.append(boxes)

    contact_points = []
    trajectories = []
    for boxes in bboxes: 
        box = boxes[0]
        y1, x1, y2, x2 = box
        bbox_offset = 20
        y1, x1, y2, x2 = int(y1) - bbox_offset, int(x1) - bbox_offset , int(y2) + bbox_offset, int(x2) + bbox_offset

        width = y2 - y1
        height = x2 - x1
        
        diff = width - height
        if width > height:
            y1 += int(diff / np.random.uniform(1.5, 2.5))
            y2 -= int((diff / (np.random.uniform(1.5, 2.5) + diff % 2)))
        else:
            diff = height - width
            x1 += int(diff / np.random.uniform(1.5, 2.5))
            x2 -= int((diff / (np.random.uniform(1.5, 2.5) + diff % 2)))

        img = np.asarray(image_pil)
        input_img = img[x1:x2, y1:y2]
        inp_img = Image.fromarray(input_img)
        inp_img = transform(inp_img).unsqueeze(0)
        gm = GaussianMixture(n_components=3, covariance_type='diag')
        centers = []
        trajs = []
        traj_scale = 0.1
        with torch.no_grad(): 
            ic, pc = net.inference(inp_img, None, None)
            pc = pc.cpu().numpy()
            ic = ic.cpu().numpy()
            i = 0
            w, h = input_img.shape[:2]
            sm = pc[i, 0]*np.array([h, w])
            centers.append(sm)
            trajs.append(ic[0, 2:])
        gm.fit(np.vstack(centers))
        cp, indx = gm.sample(50)
        x2, y2 = np.vstack(trajs)[np.random.choice(len(trajs))]
        dx, dy = np.array([x2, y2])*np.array([h, w]) + np.random.randn(2)*traj_scale
        scale = 40/max(abs(dx), abs(dy))
        adjusted_cp = np.array([y1, x1]) + cp
        contact_points.append(adjusted_cp)
        trajectories.append([x2, y2, dx, dy])
    

    original_img = np.asarray(image_pil)
    hmap = compute_heatmap(np.vstack(contact_points), (original_img.shape[1],original_img.shape[0]), k_ratio = 6)
    hmap = (hmap * 255).astype(np.uint8)
    hmap = cv2.applyColorMap(hmap, colormap=cv2.COLORMAP_JET)
    overlay = (0.6*original_img +  0.4 *hmap).astype(np.uint8)
    plt.imshow(overlay)
    for i, cp in enumerate(contact_points):
        x2, y2, dx, dy = trajectories[i]
        scale = 60/max(abs(dx), abs(dy))
        x, y = cp[:, 0] , cp[:, 1]
        plt.arrow(int(np.mean(x)), int(np.mean(y)), scale*dx, -scale*dy, color='white', linewidth=2.5, head_width=12)


    plt.axis('off')
    img_buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    im = Image.open(img_buf)
    return im