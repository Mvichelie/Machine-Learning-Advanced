import os
import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn as nn

#function to load the file list from a text file (generated by split_data)
def load_file_list(file_path):
    file_list = []
    with open(file_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            file_list.append(stripped_line)
    return file_list

# dataset class for loading images and corresponding labels with resolution and style
class OCRDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        line = self.file_list[idx].strip()
        img_path, resolution, style = line.split(',')
        img_path = img_path.strip()
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label = self.get_label_from_path(img_path)
        return image, label, resolution, style

    def get_label_from_path(self, path):
        parts = path.split(os.sep)
        if parts[-3] in ['200', '300', '400']:
            character_folder = parts[-4]
        else:
            character_folder = parts[-3]
        label = int(character_folder)
        return label

#model architecture for OCR with Conv2D and Linear layers
class OCRModel(nn.Module):
    def __init__(self, num_classes, img_dims):
        super(OCRModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (img_dims[0]//2) * (img_dims[1]//2), 128)  
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# evaluation function
def evaluate_model(val_loader, model, device):
    model.eval()
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels, resolution, style in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

#main functions
def main():
    parser = argparse.ArgumentParser(description="Evaluate an OCR model with resolution and style handling.")
    parser.add_argument('--val_file', type=str, required=True, help="Path to the validation file list.")
    parser.add_argument('--output_dir', type=str, default='./output', help="Directory where the model is saved.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--img_height', type=int, default=64, help="Height of input images.")
    parser.add_argument('--img_width', type=int, default=64, help="Width of input images.")

    args = parser.parse_args()

    val_files = load_file_list(args.val_file)

    img_dims = (args.img_height, args.img_width)

    transform = transforms.Compose([
        transforms.Resize(img_dims),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_dataset = OCRDataset(val_files, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model_path = os.path.join(args.output_dir, 'ocr_model.pth')

    # loading  the model state dictionary and retrieving the num_classes from the model itself
    checkpoint = torch.load(model_path, weights_only=True)
     # extracting num_classes from the loaded model
    num_classes = checkpoint['fc2.weight'].shape[0] 

    model = OCRModel(num_classes=num_classes, img_dims=img_dims)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(checkpoint)

    evaluate_model(val_loader, model, device)

if __name__ == "__main__":
    main()
