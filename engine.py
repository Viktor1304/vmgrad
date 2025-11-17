import math
from enum import Enum
from typing import override

from graphviz import Digraph
import sys

sys.setrecursionlimit(10000)


class Operations(Enum):
    EMTPY = ""
    ADD = "+"
    SUBSTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    TANH = "tanh"
    RELU = "relu"
    EXP = "exp"
    POW = "pow"
    LOG = "log"


class Value:
    def __init__(
        self,
        data: float,
        label: str = "",
        _children: set["Value"] | None = None,
        _op: Operations = Operations.EMTPY,
    ) -> None:
        self.data: float = data
        self.grad: float = 0.0
        self._backward = lambda: None
        if not _children:
            self._prev: set[Value] = set()
        else:
            self._prev = _children
        self._op: Operations = _op
        self.label: str = label

    def get_children(self) -> set[Value]:
        return self._prev

    def get_operation(self) -> str:
        return self._op.value

    @override
    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f})"

    def __add__(self, other: "Value | float | int") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        output: Value = Value(
            self.data + other.data, _children=set([self, other]), _op=Operations.ADD
        )

        def _backward() -> None:
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad

        output._backward = _backward

        return output

    def __mul__(self, other: "Value | int | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        output: Value = Value(
            data=self.data * other.data,
            _children=set([self, other]),
            _op=Operations.MULTIPLY,
        )

        def _backward() -> None:
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return output

    def __rmul__(self, other: "Value") -> "Value":
        return self * other

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: "Value") -> "Value":
        return self + (other * -1.0)

    def __pow__(self, other: float | int) -> Value:
        assert isinstance(other, (float, int)), "int and float powers supported"
        output: Value = Value(
            data=self.data**other,
            _children=set(
                [
                    self,
                ]
            ),
            _op=Operations.POW,
        )

        def _backward() -> None:
            self.grad += other * (self.data ** (other - 1)) * output.grad

        output._backward = _backward
        return output

    def __truediv__(self, other: "Value") -> "Value | None":
        if other.data != 0:
            return self * (other**-1)
        else:
            raise ZeroDivisionError

    def exp(self) -> Value:
        x = self.data
        output: Value = Value(
            data=math.exp(x),
            _children=set(
                [
                    self,
                ]
            ),
            _op=Operations.EXP,
        )

        def _backward() -> None:
            self.grad = output.data * output.grad

        output._backward = _backward
        return output

    def ln(self) -> Value:
        x = abs(self.data)
        output: Value = Value(
            data=math.log(x),
            _children=set(
                [
                    self,
                ]
            ),
            _op=Operations.LOG,
        )

        def _backward() -> None:
            self.grad += output.grad / x

        output._backward = _backward
        return output

    def tanh(self) -> "Value":
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        output: Value = Value(
            data=t,
            _children=set(
                [
                    self,
                ]
            ),
            _op=Operations.TANH,
        )

        def _backward() -> None:
            self.grad += (1 - t**2) * output.grad

        output._backward = _backward
        return output

    def relu(self) -> "Value":
        x = self.data
        output: Value = Value(
            data=max(0, x),
            _children=set(
                [
                    self,
                ]
            ),
            _op=Operations.RELU,
        )

        def _backward() -> None:
            if x > 0:
                self.grad += 1.0 * output.grad
            else:
                self.grad += 0.0 * output.grad

        output._backward = _backward
        return output

    @staticmethod
    def sum(arr: list["Value"]) -> "Value":
        output: Value = Value(data=0.0)
        for el in arr:
            output += el
        return output

    def backward(self):
        self.grad: float = 1.0

        topological_order: list[Value] = []
        visited: set[Value] = set()

        def build_topological_order(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v.get_children():
                    build_topological_order(child)
                topological_order.append(v)

        build_topological_order(self)

        for node in reversed(topological_order):
            node._backward()

    @staticmethod
    def trace(root: "Value") -> tuple[set[Value], set[tuple[Value, Value]]]:
        nodes: set[Value] = set()
        edges: set[tuple[Value, Value]] = set()

        def build(v: Value) -> None:
            if v not in nodes:
                nodes.add(v)
                for child in v.get_children():
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges

    def draw_dot(self) -> Digraph:
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

        nodes, edges = self.trace(self)
        for n in nodes:
            uid = str(id(n))
            dot.node(
                name=uid,
                label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
                shape="record",
            )
            if n.get_operation() != Operations.EMTPY.value:
                dot.node(name=uid + n.get_operation(), label=n.get_operation())
                dot.edge(uid + n.get_operation(), uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2.get_operation())

        dot.render(directory="output", view=True)
        return dot
