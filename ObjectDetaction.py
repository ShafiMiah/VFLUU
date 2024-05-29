#region libary and import
import torch
from torch.types import Device
from torchvision import datasets, transforms
import flwr as fl
import torch.nn as nn
from ultralytics import YOLO
import torch.optim as optim
from tqdm import tqdm
from multiprocessing import Process, freeze_support
from torch.utils.data import DataLoader
#endregion

# if __name__ == '__main__':
#     freeze_support()
#region global variable
train_image_location = 'C:\\Shafi Personal\\Study\\Masters Thesis\\Thesis Project\\Implementation\\Test\\VFLTest\\VFLYOLO8\\datasets\\coco\\images\\train2017'
train_image_annotation_file = 'C:\\Shafi Personal\\Study\\Masters Thesis\\Thesis Project\\Implementation\\Test\\VFLTest\\VFLYOLO8\\COCODataSet\\ExtractDataset\\annotations\\instances_train2017.json'
val_image_location = 'C:\\Shafi Personal\\Study\\Masters Thesis\\Thesis Project\\Implementation\\Test\\VFLTest\\VFLYOLO8\\datasets\\coco\\images\\val2017'
val_image_annotation_file = 'C:\\Shafi Personal\\Study\\Masters Thesis\\Thesis Project\\Implementation\\Test\\VFLTest\\VFLYOLO8\\COCODataSet\\ExtractDataset\\annotations\\instances_val2017.json' 
#endregion
NUM_EPOCHS = 1  # Number of epochs to train
BATCH_SIZE = 4
IMG_SIZE = (640, 640)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#region Load the COCO dataset

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def get_coco_dataloader(root, annFile, transform, batch_size):
    dataset = datasets.CocoDetection(root=root, annFile=annFile, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    return dataloader
train_loader = get_coco_dataloader(train_image_location, train_image_annotation_file, transform, BATCH_SIZE)
val_loader = get_coco_dataloader(val_image_location, val_image_annotation_file, transform, BATCH_SIZE)


#endregion

model = YOLO('yolov8n.pt').to(DEVICE)
# Loss function (Assuming YOLOv8 returns a dictionary with 'loss' key)
criterion = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training Function
def train():
    model.train()
    for epoch in range(NUM_EPOCHS):
        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs['loss'], targets)  # Assuming 'loss' is the key for the loss value
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}")
        #Evaluate the model
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to(DEVICE) for image in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                outputs = model(images)
    # Save the trained model
    torch.save(model.state_dict(), 'trained_yolov8_model.pth')

if __name__ == "__main__":
    train()

x='finish'




