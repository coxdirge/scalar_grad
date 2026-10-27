import math
from typing import Self

class Value:
    def __init__(self, data, _children=(), _op="") -> None:
        self.data = float(data)
        self._prev = tuple(_children)
        self._op = _op
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other) -> Self:
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            _children=(self, other),
            _op="+",
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other) -> Self:
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            _children=(self, other),
            _op="*",
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))

        out = Value(
            self.data ** exponent,
            _children=(self,),
            _op=f"**{exponent}",
        )

        def _backward():
            self.grad += (
                exponent * self.data ** (exponent - 1) * out.grad
            )

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def tanh(self):
        t = math.tanh(self.data)

        out = Value(
            t,
            _children=(self,),
            _op="tanh",
        )

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def dfs_topo(cur):
            if cur in visited:
                return

            visited.add(cur)
            for p in cur._prev:
                dfs_topo(p)
            topo.append(cur)

        self.grad = 1
        dfs_topo(self)

        for node in reversed(topo):
            node._backward()
