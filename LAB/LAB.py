# from Deploy_FAS_on_device import mobie
# from Data_agumentation import count_images_in_sub_datasets
# import os
#
# if __name__ == '__main__':
#     # Đường dẫn dữ liệu
#     dataset_dir = r"C:\Users\GIGABYTE\PycharmProjects\Paper-FAS\Dataset\Dataset-used"
#     train_dir = "C:\\Users\\GIGABYTE\\PycharmProjects\\Paper-FAS\\Dataset\\Dataset-used\\train"
#     val_dir = "C:\\Users\\GIGABYTE\\PycharmProjects\\Paper-FAS\\Dataset\\Dataset-used\\valid"
#     test_dir = "C:\\Users\\GIGABYTE\\PycharmProjects\\Paper-FAS\\Dataset\\Dataset-used\\test"
#     # kiem tra datagument
#
#     # Đếm số lượng ảnh
#     image_counts = count_images_in_sub_datasets(dataset_dir)
#     # Tính tổng số lượng ảnh
#     total_images = sum(image_counts.values())
#     print("Total number of images in dataset:", total_images)
#     # so luong nhan
#     num_classes = len(image_counts)
#     print("Number of classes in dataset:", num_classes)
