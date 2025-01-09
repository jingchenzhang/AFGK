# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torch import nn
import numpy as np
from skimage.transform import resize
from torchvision.utils import make_grid
import torch.nn.functional as F
# import neural_renderer  as nr

# import sys
# sys.path.append(".")

# from geometry import convert_to_full_img_cam

from pytorch3d.structures.meshes import Meshes
# from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments

from pytorch3d.renderer import (
    PerspectiveCameras,
    AmbientLights,
    RasterizationSettings,
    BlendParams,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    TexturesVertex
)

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer):
        super().__init__()
        self.rasterizer = rasterizer

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        return fragments.zbuf
    
class DepthRender:
    def __init__(self, focal_length=5000., orig_size=256, output_size=56, device=torch.device('cuda')):

        self.focal_length = focal_length
        self.orig_size = orig_size
        self.output_size = output_size
        self.device = device

       
        K = np.array([[self.focal_length, 0., self.orig_size / 2.],
                      [0., self.focal_length, self.orig_size / 2.],
                      [0., 0., 1.]])

        R = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])

        t = np.array([0, 0, 5])

        if self.orig_size != 224:
            rander_scale = self.orig_size / float(224)
            K[0, 0] *= rander_scale
            K[1, 1] *= rander_scale
            K[0, 2] *= rander_scale
            K[1, 2] *= rander_scale

        self.K = torch.FloatTensor(K[None, :, :])
        self.R = torch.FloatTensor(R[None, :, :])
        self.t = torch.FloatTensor(t[None, None, :])

        camK = F.pad(self.K, (0, 1, 0, 1), "constant", 0)
        camK[:, 2, 2] = 0
        camK[:, 3, 2] = 1
        camK[:, 2, 3] = 1

        self.KK = self.K
        self.K = camK

        self.device = device

        raster_settings = RasterizationSettings(
            image_size=self.output_size,
            blur_radius=0,
            bin_size=0,
            # max_faces_per_bin=40000,
            faces_per_pixel=1,
        )
        rasterizer=MeshRasterizer( 
            raster_settings=raster_settings
        )
        self.renderer = MeshRendererWithDepth(rasterizer)

    def camera_matrix(self, cam):
        batch_size = cam.size(0)

        K = self.K.repeat(batch_size, 1, 1)
        R = self.R.repeat(batch_size, 1, 1)
        t = torch.stack([-cam[:, 1], -cam[:, 2], 2 * self.focal_length/(self.orig_size * cam[:, 0] + 1e-9)], dim=-1)

        if cam.is_cuda:
            # device_id = cam.get_device()
            K = K.to(cam.device)
            R = R.to(cam.device)
            t = t.to(cam.device)

        return K, R, t
    
    def verts2depthimg(self, verts, faces, cam):
        batch_size = verts.size(0)

        K, R, t = self.camera_matrix(cam)

        vertices = verts
        mesh = Meshes(vertices, torch.from_numpy(faces.astype(np.float32)).to(verts.device).expand(batch_size, -1, -1))

        cameras = PerspectiveCameras(device=verts.device, R=R, T=t, K=K, in_ndc=False, image_size=[(self.orig_size, self.orig_size)])

        depth = self.renderer(mesh, cameras=cameras).permute(0, 3, 1, 2)

        # -1变为0
        # depth = depth.clamp(min=0)

        depth_list = []
        for i in range(depth.shape[0]):
            # depth_batch = torch.where(depth[i]>0, torch.log(depth[i]), depth[i])
            depth_batch = depth[i]

            foreground_mask = (depth_batch >= 0)
            if depth_batch[foreground_mask].numel()!=0:

                depth_max = torch.max(depth_batch[foreground_mask])
                # 最小值
                depth_min = torch.min(depth_batch[foreground_mask])
                # 找到除最小值外的最小值
                # second_smallest = torch.where(depth_batch == torch.min(depth_batch), torch.full_like(depth_batch, float('inf')), depth_batch).min()
                
                depth_batch = depth_batch - depth_min
                depth_batch = depth_batch / (depth_max - depth_min)

                depth_batch = torch.where(depth_batch<0, torch.ones_like(depth_batch) * -1, depth_batch)

                depth_list.append(depth_batch)
            else:
                depth_list.append(depth_batch)

        return torch.stack(depth_list, dim=0)
