from engine import Value
from nn import MLP
import torch
from torchvision import datasets, transforms
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def prepare_batch(batch_images, labels):
    ys = labels.cpu().numpy().tolist()

    flat_arr = []
    flat_labels = []

    for idx, image in enumerate(batch_images):
        arr = image.cpu().numpy()[0]
        flat_arr.append(arr.reshape(-1).tolist())

        one_hot = [0.0] * 10
        one_hot[ys[idx]] = 1.0
        flat_labels.append(one_hot)

    return flat_arr, flat_labels

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

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
    lr = 1e-3
    epochs = 10

    with ProcessPoolExecutor() as executor:
        for epoch in range(epochs):
            total_loss = 0.0
            futures = []
            batches_processed = 0

            for batch_images, labels in trainloader:
                futures.append(executor.submit(prepare_batch, batch_images, labels))
                batches_processed += 1

            for f in futures:
                flat_arr, flat_labels = f.result()

                loss = mlp.forward(flat_arr, flat_labels, lr=lr)[0]
                total_loss += loss.data


            print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss / len(trainloader)}")

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

            img_res -= 1

    print(f"{correct * 1.0 / total:.2f}")
