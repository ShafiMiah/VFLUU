#Bismillahir rahmanir rahim
#region
#endregion
#region Library
import multiprocessing
import time
from collections import OrderedDict
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader, random_split
#this is for object detection
from torchvision import datasets, transforms
import matplotlib.patches as patches
from ultralytics import YOLO
from FeatureExtractorClient import FeatureExtractorClient
from ObjectDetectionClient import DetectionHeadClient
from SplitYOLOModel import DetectionHead, FeatureExtractor
#endregion
#region Global variables
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLIENTS = 500
BATCH_SIZE = 4
#endregion
#region load dataset

#region coco dataset for object detection
train_image_location = 'C:\\Shafi Personal\\Study\\Masters Thesis\\Thesis Project\\Implementation\\Test\\VFLTest\\VFLYOLO8\\datasets\\coco\\images\\train2017'
train_image_annotation_file = 'C:\\Shafi Personal\\Study\\Masters Thesis\\Thesis Project\\Implementation\\Test\\VFLTest\\VFLYOLO8\\COCODataSet\\ExtractDataset\\annotations\\instances_train2017.json'
val_image_location = 'C:\\Shafi Personal\\Study\\Masters Thesis\\Thesis Project\\Implementation\\Test\\VFLTest\\VFLYOLO8\\datasets\\coco\\images\\test2017'
val_image_annotation_file = 'C:\\Shafi Personal\\Study\\Masters Thesis\\Thesis Project\\Implementation\\Test\\VFLTest\\VFLYOLO8\\COCODataSet\\ExtractDataset\\annotations\\instances_val2017.json' 

IMG_SIZE = (640, 640)
NUM_EPOCHS = 10  # Number of epochs to train
# VAL_SPLIT = 0.1  # Validation split ratio
TEST_SPLIT = 0.1  # Test split ratio

def get_coco_dataset(root, annFile, transform, batch_size):
    dataset = datasets.CocoDetection(root=root, annFile=annFile, transform=transform)
    return dataset
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets
def loadCoco_datasets(num_clients: int):
    transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
    dataset = get_coco_dataset(train_image_location, train_image_annotation_file, transform, BATCH_SIZE) 
    # Split dataset into train, validation, and test
    test_size_temp = int(len(dataset) * TEST_SPLIT)
    # val_size = int(len(dataset) * VAL_SPLIT)
    train_sizeTemp = int((len(dataset) - test_size_temp )/num_clients)
    train_size = train_sizeTemp *  num_clients
    test_size = len(dataset) - train_size
     # val_size = int(len(dataset) * VAL_SPLIT)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    client_datasets = random_split(train_dataset, [train_size // num_clients for _ in range(num_clients)])
    # train_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x))) for ds in client_datasets]
    # val_set = get_coco_dataset(val_image_location, val_image_annotation_file, transform, BATCH_SIZE)
    # partition_size = len(training_set) // num_clients
    # lengths = [partition_size] * num_clients
    # datasets = random_split(training_set, lengths, torch.Generator().manual_seed(42))
    # # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    for ds in client_datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
         # DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate_fn))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn= collate_fn))
    testloader =DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn = collate_fn) 
    return trainloaders, valloaders, testloader
#endregion

#trainloaders, valloaders, testloader = loadCoco_datasets(NUM_CLIENTS)
trainloaders, valloaders, testloader = loadCoco_datasets(NUM_CLIENTS)
#endregion
#region test and view object detection images
# def show_image_with_boxes(image, targets):
#     fig, ax = plt.subplots(1)
#     ax.imshow(image.permute(1, 2, 0)) 

#     for target in targets:
#         bbox = target['bbox']
#         rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)

#     plt.show()

# images, targets = next(iter(trainloaders[0]))
# # Display the first image in the batch along with its bounding boxes
# show_image_with_boxes(images[0], targets[0])
#endregion test

#region train and test model
#object detection train
def ObjectDetectionTrain(model, train_loader, optimizer, criterion, device):
    model.train()
    # for epoch in range(NUM_EPOCHS):
    for images, targets in train_loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        results = model(images)
        loss = results['loss']  # Assuming 'loss' is the key for the loss value
            
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f" Loss: {loss.item()}")
        # print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}")

# Evaluation Function
def ObjectDetectionEvaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            results = model(images)
            loss = results['loss']  # Assuming 'loss' is the key for the loss value
            
            total_loss += loss.item()
            total_samples += len(targets)

    return total_loss / total_samples

#endregion
#region train the model test (single client)

def SimualtionObjectDetectionTrain():
    #model = YOLO('yolov8s.pt').to(DEVICE)
    model = YOLO('yolov8.yaml').to(DEVICE)
    criterion = torch.nn.MSELoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainloader = trainloaders[0]
    valloader = valloaders[0]
    for epoch in range(5):
        ObjectDetectionTrain(model, trainloader, optimizer,criterion,DEVICE)
        loss= ObjectDetectionEvaluate(model, valloader)
        print(f"Epoch {epoch+1}: validation loss {loss}")

    loss =ObjectDetectionEvaluate(model, testloader) 
    print(f"Final test set performance:\n\tloss {loss}")
#endregion

#region Flower client classification

#flower client engine and start client
def client_fn_feature_extraction(cid: str) -> FeatureExtractorClient:
  
    # Load model
    print("FeatureExtractorClient calling")
    yolo_model = YOLO('C:\\Shafi Personal\\Study\\Masters Thesis\\Thesis Project\\Implementation\\VFL\\VFLUU\\yolov8s.pt')  # Load YOLOv5 model
    feature_extractor = FeatureExtractor(yolo_model)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    # Create a  single Flower client representing a single organization
    return FeatureExtractorClient(feature_extractor, trainloader, valloader).to_client()

def client_fn_detection(cid: str) -> DetectionHeadClient:
  
    # Load model
    print("DetectionHeadClient calling")
    yolo_model = YOLO('C:\\Shafi Personal\\Study\\Masters Thesis\\Thesis Project\\Implementation\\VFL\\VFLUU\\yolov8s.pt')  # Load YOLOv5 model
    detection_head  = DetectionHead(yolo_model)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    # Create a  single Flower client representing a single organization
    return DetectionHeadClient(detection_head, trainloader, valloader).to_client()

def start_client_featureextraction(cid: str):
    fl.client.start_numpy_client(server_address="localhost:8080", client= client_fn_feature_extraction(str(cid)))
def start_client_objectdetection(cid: str):
    fl.client.start_numpy_client(server_address="localhost:8081", client= client_fn_detection(str(cid)))    
#endregion

#region server side strategy
    #flower server
def start_server():
    print("Server calling")
    fl.server.start_server(
                           server_address = "localhost:8080",
                           # config = {"num_rounds": 3 },
                           # strategy = fl.server.strategy.FedAvg(
                           #      fraction_fit = 0.5,
                           #      fraction_eval = 0.5,
                           #      min_fit_clients = 2,
                           #      min_eval_clients = 2,
                           #      min_available_clients = NUM_CLIENTS,
                           # )
                           strategy = fl.server.strategy.FedAvg(
                                        min_fit_clients=2,
                                        min_eval_clients=2,
                                        min_available_clients=2,
                                    )
                           )

 
   
def SimulationStrategy():
    model = YOLO('yolov8s.pt')
    model.save('yolov8s.pt')
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #         # Submit tasks to the executor
    #         future1 = executor.submit(start_server())
    #         future2 = executor.submit( start_client_featureextraction(str(0)))
    #         future3 = executor.submit(start_client_featureextraction(str(0)))
    # start_server()        
    # start_client_featureextraction(str(0))
    # start_client_featureextraction(str(1))
    server_process = multiprocessing.Process(target=start_server)
    extraction_process = multiprocessing.Process(target=start_client_featureextraction(str(0)))
    detection_process = multiprocessing.Process(target=start_client_objectdetection(str(0)))
    server_process.start()
    extraction_process.start()
    detection_process.start()

    server_process.join()
    extraction_process.join()
    detection_process.join()
    # for i in range(NUM_CLIENTS):
    #     start_client_featureextraction(str(i))
#endregion
if __name__ == "__main__":
    #This COCO dataset is crashing now, look at this later
   SimulationStrategy() 
x=1
