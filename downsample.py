import cv2
from mlib import pf, mio, glob, tqdm, Path

sr_root = r'/media/miracle/AICO/Satellite/SRBigEarth'
lr_root = pf.get_folder(r'/media/miracle/AICO/Satellite/LRBigEarth')
sr_paths = glob.glob(sr_root+r'/*.npy')
print(f'Pairs:{len(sr_paths)}')


for sr_path in tqdm.tqdm(sr_paths):
    name = Path(sr_path).name
    sr = mio.read(sr_path)
    lr = cv2.resize(sr.transpose(1,2,0),None,fx=0.5,fy=0.5).transpose(2,0,1)
    mio.dump(lr, Path(lr_root,name))