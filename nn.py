from engine import Value
import math
import random


class Neuron:
    def __init__(self, number_inputs: int) -> None:
        sigma: float = math.sqrt(2.0 / (1 + number_inputs**2))
        self.w: list[Value] = [
            Value(data=random.gauss(0.0, sigma)) for _ in range(number_inputs)
        ]
        self.b: Value = Value(data=random.gauss(0.0, sigma))

    def __call__(self, x: list[float]) -> Value:
        if len(x) != len(self.w):
            raise ValueError("Arrays x and w must have the same size")
        activation: Value = self.b
        for wi, xi in zip(self.w, x):
            activation += wi * xi
        output: Value = activation.tanh()
        # output: Value = activation.relu()
        return output

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, number_inputs: int, number_outputs: int) -> None:
        self.neurons: list[Neuron] = [
            Neuron(number_inputs) for _ in range(number_outputs)
        ]

    def __call__(self, x: list[float]) -> list[Value]:
        outputs: list[Value] = [n(x) for n in self.neurons]
        return outputs

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, number_inputs: int, number_outputs: list[int]) -> None:
        sz: list[int] = [number_inputs] + number_outputs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(number_outputs))]

    def __call__(self, x: list[float]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def forward(
        self,
        xs: list[list[float]],
        ys: list[float] | list[list[float]],
        lr: float = 1e-3,
        cross_entropy: bool = False,
        no_grad: bool = False,
        print_predictions: bool = False,
    ) -> list[Value]:
        predictions: list[list[Value]] = [self(x) for x in xs]

        if cross_entropy:
            cross_predictions = []
            for arr in predictions:
                total: Value = Value.sum([pred.exp() for pred in arr])
                new_pred = [pred.exp() / total for pred in arr]
                cross_predictions.append(new_pred)
            predictions = list(cross_predictions)
            del cross_predictions

            loss: Value = Value.sum(
                [
                    -((yout_el.ln()) * ygt_el)
                    for ygt, yout in zip(ys, predictions)
                    for yout_el, ygt_el in zip(yout, ygt)
                ]
            )
        else:
            # Squared loss
            loss: Value = Value.sum(
                [
                    (yout_el - ygt_el) ** 2
                    for ygt, yout in zip(ys, predictions)
                    for yout_el, ygt_el in zip(yout, ygt)
                ]
            )

        if print_predictions:
            return predictions[0]

        for p in self.parameters():
            p.grad = 0.0
        loss.backward()

        if no_grad:
            return [loss]

        for p in self.parameters():
            p.data += (-lr) * p.grad

        return [loss]
