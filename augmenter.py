import numpy as np
import cv2
from torchvision.transforms import functional as F
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm

class Augmenter():

    def __init__(self, crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob):
        self.crop_augmentation_prob = crop_augmentation_prob
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.low_res_augmentation_prob = low_res_augmentation_prob

        self.random_resized_crop = transforms.RandomResizedCrop(size=(112, 112),
                                                                scale=(0.2, 1.0),
                                                                ratio=(0.75, 1.3333333333333333))
        self.photometric = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)

    def augment(self, sample):

        # crop with zero padding augmentation
        if np.random.random() < self.crop_augmentation_prob:
            # RandomResizedCrop augmentation
            sample, crop_ratio = self.crop_augment(sample)

        # low resolution augmentation
        if np.random.random() < self.low_res_augmentation_prob:
            # low res augmentation
            img_np, resize_ratio = self.low_res_augmentation(np.array(sample))
            sample = Image.fromarray(img_np.astype(np.uint8))

        # photometric augmentation
        if np.random.random() < self.photometric_augmentation_prob:
            sample = self.photometric_augmentation(sample)

        return sample

    def crop_augment(self, sample):
        new = np.zeros_like(np.array(sample))
        if hasattr(F, '_get_image_size'):
            orig_W, orig_H = F._get_image_size(sample)
        else:
            # torchvision 0.11.0 and above
            orig_W, orig_H = F.get_image_size(sample)
        i, j, h, w = self.random_resized_crop.get_params(sample,
                                                         self.random_resized_crop.scale,
                                                         self.random_resized_crop.ratio)
        cropped = F.crop(sample, i, j, h, w)
        new[i:i+h,j:j+w, :] = np.array(cropped)
        sample = Image.fromarray(new.astype(np.uint8))
        crop_ratio = min(h, w) / max(orig_H, orig_W)
        return sample, crop_ratio

    def low_res_augmentation(self, img):
        # resize the image to a small size and enlarge it back
        img_shape = img.shape
        side_ratio = np.random.uniform(0.2, 1.0)
        small_side = int(side_ratio * img_shape[0])
        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation)
        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        aug_img = cv2.resize(small_img, (img_shape[1], img_shape[0]), interpolation=interpolation)

        return aug_img, side_ratio

    def photometric_augmentation(self, sample):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.photometric.get_params(self.photometric.brightness, self.photometric.contrast,
                                        self.photometric.saturation, self.photometric.hue)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                sample = F.adjust_brightness(sample, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                sample = F.adjust_contrast(sample, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                sample = F.adjust_saturation(sample, saturation_factor)

        return sample
    
def process_dataset(input_root, output_root):
    """
    处理整个数据集的函数
    :param input_root: 输入数据集根目录（如：D:\Devloper\fr\Data\CASIA-WebFace）
    :param output_root: 输出根目录（如：D:\Devloper\fr\Data\CASIA-A）
    """
    # 初始化增强器（概率参数可根据需要调整）
    augmenter = Augmenter(
        crop_augmentation_prob=0.2,
        photometric_augmentation_prob=0.2,
        low_res_augmentation_prob=0.2
    )

    # 遍历所有图像文件
    for root, dirs, files in tqdm(os.walk(os.path.join(input_root, 'imgs'))):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 构建输入输出路径
                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(output_root, relative_path)

                # 创建输出目录
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    # 处理图像
                    with Image.open(input_path).convert('RGB') as img:
                        augmented_img = augmenter.augment(img)
                        augmented_img.save(output_path)
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")
                    continue

if __name__ == "__main__":
    # 输入输出路径配置
    input_dataset = r"D:\Devloper\fr\Data\MS1MV3"
    output_dataset = r"D:\Devloper\fr\Data\MS1MV3A"

    # 执行处理
    process_dataset(input_dataset, output_dataset)
    print("Augmentation completed!")
