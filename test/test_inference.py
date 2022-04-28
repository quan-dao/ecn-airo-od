import unittest
import torch
from model.centernet import CenterNet
import cv2
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):
    def test_something(self):
        ckpt = torch.load('../model_weights/resnet18_centernet.pth')
        model = CenterNet()
        model.load_state_dict(ckpt['model'], strict=True)
        model.cuda()
        img_name = '000000000885'
        img = cv2.imread(f'../data/val2017/{img_name}.jpg')  # (H, W, 3)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        images = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 3, H, W)
        images = model.preprocess_image(images.cuda())
        out = model.inference(images)
        torch.save(out, f'pred_for_{img_name}.pth')
        for k, v in out.items():
            print(f"{k}: {v.shape}")
        person_hm = out['cls'][0, 0].cpu().numpy()
        img_ = cv2.resize(img, (person_hm.shape[1], person_hm.shape[1]), interpolation=cv2.INTER_CUBIC)
        fig, ax = plt.subplots()
        ax.imshow(img_[..., ::-1])
        ax.imshow(person_hm, alpha=0.7)
        plt.show()

    def test_ckpt(self):
        ckpt = torch.load('../model_weights/resnet18_centernet.pth')
        model_weights = ckpt['model']
        for layer_name in model_weights.keys():
            print(layer_name)

    def test_load_img(self):
        img = cv2.imread('../data/val2017/000000000885.jpg')
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        print(f"img: {img.shape}")
        fig, axe = plt.subplots()
        axe.imshow(img[..., ::-1])
        plt.show()


if __name__ == '__main__':
    unittest.main()
