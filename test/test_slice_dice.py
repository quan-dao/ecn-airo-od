import unittest
import torch


class MyTestCase(unittest.TestCase):
    def test_something(self):
        B, H, W = 2, 2, 3
        a = torch.rand(B, 3, H, W)
        coord_x = torch.tensor([[0, 1], [1, 2]]).long()  # (B, 2)
        coord_y = torch.tensor([[0, 0], [1, 1]]).long()  # (B, 2)
        target = []  # (B, 3, 2)
        for batch_idx in range(B):
            t_ = a[batch_idx, :, coord_y[batch_idx], coord_x[batch_idx]]  # (3, 2)
            target.append(t_)
        target = torch.stack(target, dim=0)
        print(f"target: {target.shape}")

        coord_flat = coord_y * W + coord_x  # (B, 2)
        compute = torch.gather(a.reshape(B, 3, -1), dim=2, index=coord_flat.unsqueeze(1).repeat(1, 3, 1))
        print(f"compute: {compute.shape}")
        assert torch.all(compute == target)
