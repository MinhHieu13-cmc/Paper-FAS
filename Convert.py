import sys
# Thêm thư mục cha chứa folder 'onnx2tflite'
sys.path.append(r"C:\Users\GIGABYTE\PycharmProjects\Paper-FAS\onnx2tflite")
from onnx2tflite.converter import onnx_converter
from torchvision import transforms
from PIL import Image


representative_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])
representative_dir = r"C:\Users\GIGABYTE\PycharmProjects\Paper-FAS\Dataset\train_convert_reduced"

rep_dataset = FaceDataset(root_dir=representative_dir, transform=representative_transform)

onnx_converter(
    onnx_model_path = "C:\\Users\\GIGABYTE\\PycharmProjects\\Paper-FAS\Model\\face_antispoofing.onnx",
    need_simplify = True,
    output_path = "C:\\Users\\GIGABYTE\\PycharmProjects\\Paper-FAS",
    target_formats = ['keras', 'tflite'], #or ['keras'], ['keras', 'tflite']
    weight_quant = False,
    int8_model = True, # do quantification
    int8_mean = [0.55717504 ,0.45449087 ,0.41123426], # give mean of image preprocessing
    int8_std = [0.1968333  ,0.17541456 ,0.16581647], # give std of image preprocessing
    image_root = "C:\\Users\\GIGABYTE\\PycharmProjects\\Paper-FAS\\Dataset\\Dataset-used\\train" # give image folder of train_convert
)