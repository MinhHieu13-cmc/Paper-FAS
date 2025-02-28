import os

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


