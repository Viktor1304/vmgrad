from engine import Value
from nn import MLP
import torch
from torchvision import datasets, transforms


transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms
)
testset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

mlp = MLP(28 * 28, [32, 10])
lr = 1e-2
epochs = 10

for epoch in range(epochs):
    total_loss = Value(data=0.0)

    for batch_images, labels in trainloader:
        ys = labels.cpu().detach().numpy().tolist()
        flat_arr = []
        flat_labels: list[list[float]] = []
        for idx, image in enumerate(batch_images):
            arr_image = image.cpu().detach().numpy().tolist()[0]
            flat_image = [x for arr in arr_image for x in arr]
            flat_arr.append(flat_image)

            one_hot = [0.0 for _ in range(10)]
            one_hot[ys[idx]] = 1.0
            flat_labels.append(one_hot)

        loss: Value = mlp.forward(flat_arr, flat_labels, lr=lr)[0]

        total_loss.data += loss.data

        del loss

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss.data / len(trainloader)}")


correct: int = 0
total: int = 0
for batch_images, labels in testloader:
    ys = labels.cpu().detach().numpy().tolist()
    for idx, image in enumerate(batch_images):
        arr_image = image.cpu().detach().numpy().tolist()[0]
        flat_image = [x for arr in arr_image for x in arr]

        one_hot = [0.0 for _ in range(10)]
        one_hot[ys[idx]] = 1.0
        predictions: list[Value] = mlp.forward(
            [flat_image],
            [one_hot],
            cross_entropy=True,
            no_grad=True,
            print_predictions=True,
        )

        max_val, max_idx = -1.0, -1
        for idx, val in enumerate(predictions):
            if val.data > max_val:
                max_val = val.data
                max_idx = idx

        if one_hot[max_idx] == 1.0:
            correct += 1
        total += 1

print(f"{correct * 1.0 / total:.2f}")
