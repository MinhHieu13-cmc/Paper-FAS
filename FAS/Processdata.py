from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Đường dẫn đến thư mục gốc (vd: train_dir hoặc val_dir).
        :param transform: Các hàm tiền xử lý ảnh (Resize, Normalize, Augmentation, etc.)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Lần lượt duyệt qua các lớp (folder fake và real)
        for label, folder_name in enumerate(['fake', 'real']):  # 0: fake, 1: real
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} not found!")
                continue

            # Lấy danh sách các ảnh trong từng thư mục
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                self.data.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Đọc ảnh từ đường dẫn lưu trữ
        img_path = self.data[idx]
        label = self.labels[idx]

        # Mở và xử lý ảnh
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label


class EarlyStopping:
    """
    Dừng sớm quá trình training nếu validation loss không cải thiện sau một số epoch cố định.
    """

    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
        :param patience: Số epoch không cải thiện trước khi dừng training
        :param verbose: In log nếu True
        :param delta: Sự cải thiện nhỏ nhất để cập nhật
        :param path: Đường dẫn lưu mô hình checkpoint
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Lưu model khi validation loss giảm"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



