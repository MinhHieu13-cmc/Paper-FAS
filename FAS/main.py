import torch
import torch.nn as nn
from FaceAntiSpoofingModel import FaceAntiSpoofingModel
from Processdata import EarlyStopping , FaceDataset

if __name__ == '__main__':
    # Bộ tiền xử lý ảnh (resize, normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ảnh về kích thước phù hợp
        transforms.ToTensor(),  # Chuyển đổi thành dạng tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Định nghĩa tập train và val
    train_dataset = FaceDataset(root_dir=train_dir, transform=transform)
    val_dataset = FaceDataset(root_dir=val_dir, transform=transform)

    # Tạo DataLoader cho tập train và val
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Cấu hình thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = FaceAntiSpoofingModel(num_classes=2).to(device)

    # Loss, Optimizer và Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Số epoch và EarlyStopping
    num_epochs = 50
    early_stopping = EarlyStopping(patience=5, verbose=True, path='face_antispoofing_model.pth')

    # Quá trình huấn luyện
    for epoch in range(num_epochs):
        # Phase Train
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass và cập nhật trọng số
            loss.backward()
            optimizer.step()

            # Tích lũy loss và accuracy
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Phase Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Tích lũy loss và accuracy
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val

        # In thông tin mỗi epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Scheduler giảm learning rate nếu val loss không cải thiện
        scheduler.step(val_loss)

        # Early stopping kiểm tra và lưu checkpoint
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
