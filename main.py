from batch_data_split import train_data, test_data
from model_conv import Model
from model_lin import LinModel
import utils
import torch

conv = Model(utils.input_channels)
lin = LinModel()
conv_opt = torch.optim.Adam(conv.parameters(), lr=utils.lr)
lin_opt = torch.optim.Adam(lin.parameters(), lr=utils.lr)
loss = torch.nn.CrossEntropyLoss()


def train(conv_train=True):
    if conv_train:
        for epoch in range(utils.epochs):
            for (img, label) in train_data:
                output = conv(img)
                l = loss(output, label)
                print(f"for epoch {epoch} loss is {l} ")
                l.backward()
                conv_opt.step()
                conv_opt.zero_grad()
        utils.testing(conv, test_data)

    else:
        for epoch in range(utils.epochs):
            for (img, label) in train_data:
                output = lin(img)
                l = loss(output, label)
                print(f"for epoch {epoch} loss is {l}")
                l.backward()
                lin_opt.step()
                lin_opt.zero_grad()
        # utils.testing(lin, test_data)


train(conv_train=True)
# train(conv_train=False)
