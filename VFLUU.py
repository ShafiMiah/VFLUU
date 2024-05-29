#Bismillahir rahmanir rahim
#region
#endregion
#region Library
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

from CNN import CNNNet
#endregion
#region Global variables
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda')
NUM_CLIENTS = 10
BATCH_SIZE = 32
# BATCH_SIZE = 4
#endregion
#region load dataset
#region cifar dataset for classification
def loadCifar_datasets(batch_size: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8, seed=42)
        trainloaders.append(DataLoader(partition["train"], batch_size=batch_size))
        valloaders.append(DataLoader(partition["test"], batch_size=batch_size))
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader
#endregion
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
trainloaders, valloaders, testloader = loadCifar_datasets(BATCH_SIZE)
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
#region view classification images
# batch = next(iter(trainloaders[0]))
# images, labels = batch["img"], batch["label"]
# # Reshape and convert images to a NumPy array
# # matplotlib requires images with the shape (height, width, 3)
# images = images.permute(0, 2, 3, 1).numpy()
# # Denormalize
# images = images / 2 + 0.5

# # Create a figure and a grid of subplots
# fig, axs = plt.subplots(4, 8, figsize=(12, 6))

# # Loop over the images and plot them
# for i, ax in enumerate(axs.flat):
#     ax.imshow(images[i])
#     ax.set_title(trainloaders[0].dataset.features["label"].int2str([labels[i]])[0])
#     ax.axis("off")

# # Show the plot
# fig.tight_layout()
# plt.show()
#endregion
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
#classification train
def ClassificationTrain(net, trainloader, epochs: int, verbose=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def ClassificationTest(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
#endregion
#region train the model test (single client)

def SimualtionObjectDetectionTrain():
    #model = YOLO('yolov8n.pt').to(DEVICE)
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

def SimualtionClassificationTrain():
    trainloader = trainloaders[0]
    valloader = valloaders[0]
    net = CNNNet().to(DEVICE)

    for epoch in range(5):
        ClassificationTrain(net, trainloader, 1)
        loss, accuracy = ClassificationTest(net, valloader)
        print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

    loss, accuracy = ClassificationTest(net, testloader)
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
#endregion
#region Flower client-server interaction
#parameters received from the server and update client
def set_parameters_classification(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
#get the updated model parameters from the local model    
def get_parameters_classification(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]  


#endregion
#region Flower client
class FlowerClientClassification(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters_classification(self.net)

    def fit(self, parameters, config):
        set_parameters_classification(self.net, parameters)
        ClassificationTrain(self.net, self.trainloader, epochs=1)
        return get_parameters_classification(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters_classification(self.net, parameters)
        loss, accuracy = ClassificationTest(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
#flower client engine
def client_fn(cid: str) -> FlowerClientClassification:
  
    # Load model
    net = CNNNet().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    # Create a  single Flower client representing a single organization
    return FlowerClientClassification(net, trainloader, valloader).to_client()
#endregion

#region server side strategy
    #flower server
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
def start_server():
    fl.server.start_server(
    server_address="localhost:8080",
    config={"num_rounds": 3},
    strategy=fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_eval=0.5,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=NUM_CLIENTS,
        # evaluate_metrics_aggregation_fn=weighted_average
    )
    )
def start_client(cid: str):
    fl.client.start_numpy_client(server_address="localhost:8080", client= client_fn(str(cid)))

    
def SimulationStrategy():
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=1.0,  # Sample 100% of available clients for training
    #     # fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    #     min_fit_clients=10,  # Never sample less than 10 clients for training
    #     min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    #     min_available_clients=10,  # Wait until all 10 clients are available
    # )
    start_server()
    for i in range(NUM_CLIENTS):
        start_client(str(i))
#endregion
if __name__ == "__main__":
    #This COCO dataset is crashing now, look at this later
    #SimualtionObjectDetectionTrain()
    #Classification task
   #SimualtionClassificationTrain()
   SimulationStrategy() 
x=1
