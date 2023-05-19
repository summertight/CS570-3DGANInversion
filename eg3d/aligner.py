from kornia.geometry.transform import warp_affine
from torch.linalg import lstsq as solver
import torch

def align_and_crop_with_5points_tensor(img, landmark, size=224):
        assert len(landmark.shape) == 3
        #src_pts = torch.tensor(landmark, dtype=torch.float64).unsqueeze(0)
        B = landmark.shape[0]

        REFERENCE_FACIAL_POINTS = [
                [30.29459953,  51.69630051],
                [65.53179932,  51.50139999],
                [48.02519989,  71.73660278],
                [33.54930115,  92.3655014],
                [62.72990036,  92.20410156]
                ]

        REFERENCE_FACIAL_POINTS = torch.tensor(REFERENCE_FACIAL_POINTS, dtype=torch.float64)
        REFERENCE_FACIAL_POINTS[..., 0] += 8
        REFERENCE_FACIAL_POINTS *= size / 112.0


        dst_pts = REFERENCE_FACIAL_POINTS.unsqueeze(0).repeat(B,1,1)
        # align dst to src

        tfm = torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).repeat(B,1,1)
        n_pts = dst_pts.shape[1]
        ones = torch.ones((B, n_pts, 1), dtype=torch.float64)
        src_pts_ = torch.cat([landmark, ones],axis=-1)
        dst_pts_ = torch.cat([dst_pts, ones],axis=-1)

        A, res, rank, s = solver(src_pts_, dst_pts_)

        #if rank == 3:
        breakpoint()
        if True:
                tfm = A[..., :2].T
                #tfm = torch.tensor([[A[..., 0, 0], A[..., 1, 0], A[..., 2, 0]],
                #        [A[..., 0, 1], A[..., 1, 1], A[..., 2, 1]]],dtype=torch.float32).unsqueeze(0)
        elif rank==2:
                tfm = torch.tensor([[A[..., 0, 0], A[..., 1, 0], 0],
                        [A[..., 0, 1], A[..., 1, 1], 0]],dtype=torch.float32).unsqueeze(0)

        outs = warp_affine(img.float(), tfm, [size,size], padding_mode="zeros")

        return outs