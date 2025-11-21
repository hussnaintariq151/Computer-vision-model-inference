import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import LeNet

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Load model
model = LeNet().to(device)
model.load_state_dict(torch.load("models/lenet_mnist.pth", map_location=device))
model.eval()

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


