from torch import nn


class CNNAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
        )

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )

        self.conv1T = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=(1,1)
        )

        self.conv2T = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=(1,0)
        )

        self.conv3T = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=(1,1)
        )

        self.conv4T = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=(1,1)
        )

    
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(128 * 4 * 3, bottleneck_dim)
        self.softmax = nn.Softmax(dim=1)

        self.linear2 = nn.Linear(bottleneck_dim, 128 * 4 * 3)
        self.unflatten = nn.Unflatten(1, (128, 4, 3))

        self.relu = nn.ReLU()

    def forward(self, input_data):
        # Encode
        out = self.relu(self.conv1(input_data))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))

        out = self.flatten(out)
        out = self.linear1(out)
        out = self.softmax(out)

        embedding = out

        # Decode
        out = self.linear2(out)
        out = self.unflatten(out)

        out = self.relu(self.conv1T(out))
        out = self.relu(self.conv2T(out))
        out = self.relu(self.conv3T(out))
        out = self.relu(self.conv4T(out))    

        return out, embedding


if __name__ == "__main__":
    cnn = CNNAutoencoder()
