from torch import nn


class CNNAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
        )

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
        )

        self.conv1T = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=2,
        )

        self.conv2T = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=2,
        )

        self.conv3T = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=2,
        )

        self.conv4T = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=2,
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(kernel_size=2)
    
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(128 * 5 * 4, bottleneck_dim)
        # self.linear1 = nn.Linear(128 * 5 * 2, 10)
        self.softmax = nn.Softmax(dim=1)

        # self.linear2 = nn.Linear(10, 128 * 5 * 2)
        self.linear2 = nn.Linear(bottleneck_dim, 128 * 5 * 4)
        self.unflatten = nn.Unflatten(1, (128, 5, 4))
        # self.unflatten = nn.Unflatten(1, (128, 5, 2))

        self.relu = nn.ReLU()

    def forward(self, input_data):
        # Encode
        out = self.relu(self.conv1(input_data))
        size1 = out.size()
        out, indices1 = self.maxpool(out)

        out = self.relu(self.conv2(out))
        size2 = out.size()
        out, indices2 = self.maxpool(out)

        out = self.relu(self.conv3(out))
        size3 = out.size()
        out, indices3 = self.maxpool(out)

        out = self.relu(self.conv4(out))
        size4 = out.size()
        out, indices4 = self.maxpool(out)

        out = self.flatten(out)
        out = self.linear1(out)
        out = self.softmax(out)

        embedding = out

        # Decode
        out = self.linear2(out)
        out = self.unflatten(out)

        out = self.conv1T(
            self.relu(self.maxunpool(out, indices=indices4, output_size=size4))
        )
        out = self.conv2T(
            self.relu(self.maxunpool(out, indices=indices3, output_size=size3))
        )
        out = self.conv3T(
            self.relu(self.maxunpool(out, indices=indices2, output_size=size2))
        )
        out = self.conv4T(
            self.relu(self.maxunpool(out, indices=indices1, output_size=size1))
        )

        return out, embedding


if __name__ == "__main__":
    cnn = CNNAutoencoder()
