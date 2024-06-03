import flwr as fl
import torch
import torchvision.datasets as datasets
import torch.nn as nn
# Define Flower client for Feature Extractor
class FeatureExtractorClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for images, targets in self.trainloader:
            self.optimizer.zero_grad()
            features = self.model(images)
             # Note: need to adjust this as per my task
            loss = self.criterion(features, targets) 
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, targets in self.testloader:
                outputs = self.model(images)
                 # Note: need to adjust this as per my task
                loss = self.criterion(outputs, targets) 
                correct += (outputs.argmax(1) == targets).sum().item()
                total += targets.size(0)
        return float(correct) / total, len(self.testloader.dataset), {}
