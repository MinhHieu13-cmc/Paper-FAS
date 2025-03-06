import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Hàm đếm tổng số lượng ảnh trong mỗi tệp
def count_images_in_sub_datasets(dataset_dir):
    sub_datasets = ['train', 'valid', 'test']
    counts = {}

    for sub in sub_datasets:
        sub_dir = os.path.join(dataset_dir, sub)
        if os.path.exists(sub_dir):
            # Đếm tổng số ảnh trong tất cả các thư mục con
            total_images = sum([
                len(files) for _, _, files in os.walk(sub_dir)
            ])
            counts[sub] = total_images
        else:
            counts[sub] = 0

    return counts

# Lớp xử lý dữ liệu đầu vào cho backbone
class DataProcessor:
    def __init__(self, image_size=(224, 224), augment=False):
        self.image_size = image_size
        self.augment = augment
        self.datagen = self._get_data_generator()

    def _get_data_generator(self):
        if self.augment:
            return ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            return ImageDataGenerator()

    def load_and_preprocess_image(self, image_path):
        """Đọc ảnh, resize và chuẩn hóa pixel về [0, 1]"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image / 255.0
        return image

    def augment_image(self, image):
        """Áp dụng augmentation lên ảnh"""
        image = np.expand_dims(image, axis=0)
        return next(self.datagen.flow(image, batch_size=1))[0]

    def process_batch(self, image_paths):
        """Xử lý một batch hình ảnh"""
        images = [self.load_and_preprocess_image(img) for img in image_paths]
        if self.augment:
            images = [self.augment_image(img) for img in images]
        return np.array(images)