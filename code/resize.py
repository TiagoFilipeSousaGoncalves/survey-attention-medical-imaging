import os
from PIL import Image
from tqdm import tqdm
import numpy as np

DATAPATH = "/BARRACUDA8T/DATASETS/APTOS2019/train_images/"
NEWPATH = "/BARRACUDA8T/DATASETS/APTOS2019/train_resized/"
NEW_HEIGHT = 512

if not os.path.exists(NEWPATH):
    os.makedirs(NEWPATH)
    
for f in tqdm(os.listdir(DATAPATH)):
    if(f.endswith(".jpg") or f.endswith('.png')):
        img = Image.open(os.path.join(DATAPATH, f))
        w, h = img.size
        ratio = w / h
        new_w = int(np.ceil(NEW_HEIGHT * ratio))
        new_img = img.resize((new_w, NEW_HEIGHT), Image.ANTIALIAS)
        new_img.save(os.path.join(NEWPATH, f))
