# classifiers
import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2**15, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.pool4(torch.relu(self.conv4(x)))

        x = self.flatten(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        # if softmax:
        #    x = self.softmax(x)
        return x


class EnsembleModel(nn.Module):
    def __init__(self, models, num_classes, device, weight_paths=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.input_transforms = []
        for i, model in enumerate(self.models):
            if weight_paths is not None and weight_paths[i] is not None:
                model.load_state_dict(torch.load(weight_paths[i], weights_only=True))
            model.to(device)
            model.eval()
            inp_channels = next(model.parameters()).shape[1]
            if inp_channels == 1:
                self.input_transforms.append(lambda x: x.mean(dim=1, keepdim=True))
            elif inp_channels == 3:
                self.input_transforms.append(lambda x: x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x)
            else:
                self.input_transforms.append(lambda x: x)
        self.classifier = nn.Linear(num_classes * len(models), num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        outputs = [model(transform(x)) for model, transform in zip(self.models, self.input_transforms)]
        x_ens = self.relu(torch.cat(outputs, dim=1))  # Shape: (16, 108)
        # x_ens = self.relu(torch.cat([model(x_rs(model, x)) for model in self.models], dim=1))
        out = self.classifier(x_ens)
        return out


def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def load_model_gpu(model, device, is_cuda):
    if is_cuda:
        model.to(device)
    return model
