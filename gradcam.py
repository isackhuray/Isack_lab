from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import timm
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class GradcamExtractor:
    def __init__(self, model_name, target_layer_idx=0):
        self.model_name = model_name
        self.target_layer_idx = target_layer_idx
        self.model = self._load_model()
        self.cam = None

    def _load_model(self):
        model = timm.create_model(self.model_name, pretrained=True).cuda()
        model.eval()
        return model

    def _img_transform(self, img):
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        img = transform(img).unsqueeze(0).cuda()
        return img

    def extract(self, img_path, output_path):
        image = Image.open(img_path)
        image = self._img_transform(image)

        if self.cam is None:
            target_layers = [self.model.stages[self.target_layer_idx]]
            self.cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=True)

        grayscale_cam = self.cam(input_tensor=image)
        grayscale_cam = grayscale_cam[0, :]

        # Grad-CAM 결과물 저장
        plt.imshow(grayscale_cam, cmap='jet')
        plt.savefig(output_path)
        print(f"Grad-CAM 이미지 저장 완료: {output_path}")

if __name__ == "__main__":
    model_name = "resnetv2_50x1_bit.goog_in21k_ft_in1k" #백본모델이름
    img_path = "/data2/aihub/train/Training/09016001/09_096_09016001_160638583265890_1.jpg" #입력이미지
    output_path = "/data1/Isack_total/gradcam_result/output_image2.jpg" #gradcam 저장 이미지
    
    gradcam_extractor = GradcamExtractor(model_name)
    gradcam_extractor.extract(img_path, output_path)
