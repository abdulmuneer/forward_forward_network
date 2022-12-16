import torch
from torch import nn


class Layer(nn.Linear):
    """Customized Linear layers.

    We add methods for training and computing goodness of a layer.
    """

    def __init__(self, in_size, out_size, threshold=2):
        super().__init__(in_size, out_size)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.optimizer = torch.optim.Adam(self.parameters())

    def update_optimiser(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        direction = x / (x.norm(keepdim=True) + 1e-4)
        layer_out = super().forward(x)
        activation = self.relu(layer_out)
        return activation

    def goodness(self, H):
        return (H**2).mean()

    def train(self, example, good_data=True):
        fwd_out = self.forward(example)
        if good_data:
            loss = torch.log(1 + torch.exp(self.threshold - self.goodness(fwd_out)))

        else:
            loss = torch.log(1 + torch.exp(self.goodness(fwd_out) - self.threshold))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return fwd_out.detach()


class ForwardForwardNet(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        self.layers = []

        # initalises the customised Linear layers
        for in_feature, out_feature in zip(layers, layers[1:]):
            self.layers.append(Layer(in_feature, out_feature))

    def update_optimiser(self, lr):
        for layer in self.layers:
            layer.update_optimiser(lr=lr)

    def predict(self, x):
        predictions = []
        best_activation, best_activation_class = None, None
        for i in range(10):
            label = torch.ones(size=[x.shape[0]]) * i
            example = club_labels(x, label)
            layer_data = example
            for layers in self.layers:
                layer_data = layers.forward(layer_data)
            prediction = (layer_data**2).mean(axis=1)
            predictions.append(prediction)
        return torch.vstack(predictions).T.argmax(axis=1)

    def train(self, example, good_data=True):
        """Trains all layers sequentially.

        Since we are passing the layer output at every step to the next layer.
        Another way could be to partially train one layer to maximise its activation
        before passing its output to the next layer. This approach also can be explored.
        """
        layer_training_data = example
        for layers in self.layers:
            layer_training_data = layers.train(layer_training_data, good_data=good_data)
