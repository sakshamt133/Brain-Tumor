import torch

batch_size = 16
epochs = 5
lr = 0.001
input_channels = 1


def testing(model, test_data):
    with torch.no_grad():
        sume = 0
        tot = 0
        for img, test_label in test_data:
            yhat = model(img)
            out = yhat.clone().detach()
            real = []
            for i, val in enumerate(out):
                real.append(torch.argmax(val))

            for i, val in enumerate(real):
                if val == test_label[i].item():
                    sume += 1
                tot += 1
        print(f"accuracy is {sume/tot}")
