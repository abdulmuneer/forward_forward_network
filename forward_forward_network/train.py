import torch
import torchvision
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from forward_forward import ForwardForwardNet


def get_mnist_dataloaders(
    data_location="data/mnist/", train_batch=1024, val_batch=1024
):
    # loading MNIST
    # MNIST images are PIL objects
    # need to convert them to tensors as data loader expects tensors or convertible dtypes in dataset.
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0, std=1),
            torch.flatten,
        ]
    )
    train_data = torchvision.datasets.MNIST(
        data_location, download=True, transform=transforms
    )
    valid_data = torchvision.datasets.MNIST(
        data_location, download=True, train=False, transform=transforms
    )
    train_dl = DataLoader(train_data, batch_size=train_batch)
    valid_dl = DataLoader(valid_data, batch_size=val_batch)
    return train_dl, valid_dl


# prepare wrong_labels for "bad" data
def generate_bad_examples(sample_data):
    data, good_label = sample_data

    for i in range(1, 10):
        # labels cycle through [0-9] digits, starting from their original value
        bad_label = (good_label + i) % 10
        yield club_labels(data, bad_label)


def generate_good_examples(sample_data):
    data, label = sample_data
    yield club_labels(data, label)


def club_labels(data, label):
    """Appends the 1-hot encoded labels to the flattened input tensor.

    The paper talks about inserting the labels in the border that are
    available in MNIST. The borders in MNIST were provided to play nicely with convlution.
    Since this network uses the images reshaped to 1D, I chose to add the label at the end.
    """
    try:
        one_hot_labels = nn.functional.one_hot(
            label.to(torch.int64), num_classes=10
        ).to(data.dtype)
    except RuntimeError:
        import pdb

        pdb.set_trace()
    return torch.hstack((data, one_hot_labels))


def train(num_epoch=1, train_batch=1024, val_batch=1024, lr=1e-4):
    """The entry point for training the model."""
    # initialize Model
    ffn = ForwardForwardNet([794, 500, 100])
    ffn.update_optimiser(lr)

    # load training and validation data
    train_dl, val_dl = get_mnist_dataloaders(
        train_batch=train_batch, val_batch=val_batch
    )

    for epoch in range(num_epoch):
        print(f"Epoch: {epoch}")
        print("Training")
        for i, sample_data in tqdm.tqdm(enumerate(train_dl)):
            # train with correct label
            for good_examples in generate_good_examples(sample_data):
                ffn.train(good_examples, good_data=True)

            # train with incorrect labels
            for bad_examples in generate_bad_examples(sample_data):
                ffn.train(bad_examples, good_data=False)

        print("testing")
        mismatched_numbers = 0
        total_set = 0
        for i, sample_data in enumerate(val_dl):
            sample_d, sample_good_label = sample_data
            # prepare "good" data
            one_hot_labels = nn.functional.one_hot(sample_data[1], num_classes=10)
            new_t_data = torch.hstack((sample_data[0], one_hot_labels))
            pred = ffn.predict(sample_d)
            actual = sample_good_label
            mismatched_numbers += torch.count_nonzero(pred - actual)
            total_set += sample_good_label.shape[0]
            # print(f"Actual:\t {actual}")
            # print(f"Predicted:\t {pred}")

        mismatch_pct = (mismatched_numbers / total_set) * 100
        print(f"mismatch: {mismatch_pct} %")


if __name__ == "__main__":
    train(num_epoch=1, train_batch=1024, val_batch=1024, lr=1e-4)
