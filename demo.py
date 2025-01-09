import numpy as np
import torch
import os
import cv2
from tqdm import tqdm
from glob import glob

from lib.core.config import parse_args
from lib import get_model
from lib.renderer.renderer_img import Renderer as Renderer_img

from lib.datasets.detect_dataset import DetectDataset
from lib.models.smpl import SMPL
from lib.yolo import Yolov7

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"  # 或 "osmesa"

# Yolo model
DEVICE = 'cpu'
yolo = Yolov7(device=DEVICE, weights='data/pretrain/yolov7-e6e.pt', imgsz=1281)

# ReFit
args = ['--cfg', 'configs/config.yaml']
cfg = parse_args(args)
cfg.DEVICE = 'cpu'

model = get_model(cfg)
checkpoint = '/home/zjc/work_dir/code/refit/results/only 3d/checkpoint_best.pth.tar'
state_dict = torch.load(checkpoint, map_location=cfg.DEVICE)
_ = model.load_state_dict(state_dict['model'], strict=False)
_ = model.eval()
print('Loaded checkpoint:', checkpoint)

# Rendering
smpl = SMPL()
renderer_img = Renderer_img(smpl.faces, color=(0.9, 0.9, 0.9))

# Example image
# imgname = 'data/examples/skates.png'
imgfiles = sorted(glob('/home/zjc/work_dir/code/refit/data/output/*'))
for imgname in tqdm(imgfiles):
    img = cv2.imread(imgname)[:,:,::-1].copy()

    ### --- Detection ---
    with torch.no_grad():
        boxes = yolo(img, conf=0.50, iou=0.45)
        
    db = DetectDataset(img, boxes)
    dataloader = torch.utils.data.DataLoader(db, batch_size=8, shuffle=False, num_workers=0)

    ### --- ReFit ---
    vert_all = []
    for batch in dataloader:
        with torch.no_grad():
            out, preds = model(batch, iters=5)
            s_out = model.smpl.query(out)
            vertices = s_out.vertices

        vert = vertices
        trans = out['trans_full']
        vert_full = vert + trans
        vert_all.append(vert_full)
        
    vert_all = torch.cat(vert_all)

    ### --- Render ---
    img_render = renderer_img(vert_all, [0,0,0], img)
    new_name = os.path.basename(imgname).replace('.png', '_result.png').replace('.jpg', '_result.jpg')
    cv2.imwrite(new_name, img_render[:,:,::-1].copy())
    print('Done')
# import numpy as np
# import torch
# import os
# import cv2
# from tqdm import tqdm
# from glob import glob

# from lib.core.config import parse_args
# from lib import get_model
# from lib.renderer.renderer_img import Renderer as Renderer_img

# from lib.datasets.detect_dataset import DetectDataset
# from lib.models.smpl import SMPL
# from lib.yolo import Yolov7
# from lib.utils.geometry import perspective_projection

# import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"  # 或 "osmesa"

# # Yolo model
# DEVICE = 'cpu'
# yolo = Yolov7(device=DEVICE, weights='data/pretrain/yolov7-e6e.pt', imgsz=1281)

# # ReFit
# args = ['--cfg', 'configs/config.yaml']
# cfg = parse_args(args)
# cfg.DEVICE = 'cpu'

# model = get_model(cfg)
# checkpoint = 'data/pretrain/refit_all/checkpoint_best.pth.tar'
# state_dict = torch.load(checkpoint, map_location=cfg.DEVICE)
# _ = model.load_state_dict(state_dict['model'], strict=False)
# _ = model.eval()
# print('Loaded checkpoint:', checkpoint)

# # Rendering
# smpl = SMPL()
# renderer_img = Renderer_img(smpl.faces, color=(0.9, 0.9, 0.9, 1.0))

# # Define the project function properly
# def project(points, pred_cam, center, scale, img_focal, img_center, return_full=False):
#     trans_full = get_trans(pred_cam, center, scale, img_focal, img_center)  # Get full transform

#     # Projection in full frame image coordinate
#     points = points + trans_full  # Apply transform to the 3D points
#     points2d_full = perspective_projection(points, rotation=None, translation=None,
#                         focal_length=img_focal, camera_center=img_center)

#     # Adjust projected points to crop image coordinate
#     b = scale * 200
#     points2d = points2d_full - (center - b[:, None] / 2)[:, None, :]
#     points2d = points2d * (256/ b)[:, None, None]

#     if return_full:
#         return points2d_full, points2d
#     else:
#         return points2d

# def get_trans(pred_cam, center, scale, img_focal, img_center):
#     b = scale * 200
#     cx, cy = center[:, 0], center[:, 1]  # center of crop
#     s, tx, ty = pred_cam.unbind(-1)

#     img_cx, img_cy = img_center[:, 0], img_center[:, 1]  # center of original image

#     bs = b * s
#     tx_full = tx + 2 * (cx - img_cx) / bs
#     ty_full = ty + 2 * (cy - img_cy) / bs
#     tz_full = 2 * img_focal / bs

#     trans_full = torch.stack([tx_full, ty_full, tz_full], dim=-1)
#     trans_full = trans_full.unsqueeze(1)

#     return trans_full

# # Example image
# imgfiles = sorted(glob('/home/zjc/work_dir/code/refit/data/output/*'))
# for imgname in tqdm(imgfiles):
#     img = cv2.imread(imgname)[:, :, ::-1].copy()

#     ### --- Detection ---
#     with torch.no_grad():
#         boxes = yolo(img, conf=0.50, iou=0.45)

#     db = DetectDataset(img, boxes)
#     dataloader = torch.utils.data.DataLoader(db, batch_size=8, shuffle=False, num_workers=0)

#     ### --- ReFit ---
#     vert_all = []
#     joint_all = []  # To store the 2D projected joint positions
#     for batch in dataloader:
#         with torch.no_grad():
#             out, preds = model(batch, iters=5)
#             s_out = model.smpl.query(out)
#             vertices = s_out.vertices
#             image = batch['img']
#             center = batch['center']
#             scale = batch['scale']
#             img_focal = batch['img_focal']
#             img_center = batch['img_center']

#         vert = vertices
#         trans = out['trans_full']
#         vert_full = vert + trans
#         vert_all.append(vert_full)

#         # Get 3D joints (24 joints)
#         j3d = s_out.joints  # 3D joints (24 joints)

#         # Project 3D joints to 2D using the function
#         j2d = project(j3d, out['pred_cam'], center, scale, img_focal, img_center)

#         joint_all.append(j2d)
#     vert_all = torch.cat(vert_all)

#     ### --- Render ---
#     img_render = renderer_img(vert_all, [0, 0, 0], img)

#     # Draw 2D joints on the image
# # Draw 2D joints on the image
# # Draw 2D joints on the image
# # Assuming joint_all is in the format: [tensor([[x1, y1], [x2, y2], ..., [xn, yn]])]
# # Example for processing the joints and drawing on the image

# for joints_tensor in joint_all:
#     # Iterate over each joint in the tensor
#     for joint in joints_tensor[0]:  # `joints_tensor[0]` is the 2D coordinates (Nx2)
#         x, y = joint[0].item(), joint[1].item()  # Convert tensor values to scalar
        
#         # Draw red dots on the image at the (x, y) positions
#         cv2.circle(img_render, (int(x), int(y)), 2, (255, 0, 0), -1)  # Red color: (0, 0, 255)





#     # Save the result
#     new_name = os.path.basename(imgname).replace('.png', '_result.png').replace('.jpg', '_result.jpg')
#     cv2.imwrite(new_name, img_render[:, :, ::-1].copy())
#     print(f'Done: {new_name}')







