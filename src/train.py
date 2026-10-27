import random
from nn import MLP

random.seed(42)

# 训练数据
xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
ys = [-1.0, 1.0, 1.0, 1.0]

model = MLP(2, [4, 4, 1])

for step in range(100):
    # 1. Forward pass
    ypred = [model(x) for x in xs]

    # 2. Compute loss
    loss = sum(
        (pred - target) ** 2
        for pred, target in zip(ypred, ys)
    )

    # 3. Zero gradients
    for p in model.parameters():
        p.grad = 0.0

    # 4. Backward pass
    loss.backward()

    # 5. Gradient descent
    for p in model.parameters():
        p.data -= 0.05 * p.grad

    if step % 20 == 0:
        print(f"step {step}: loss = {loss.data:.6f}")

print("\nPredictions:")
for x in xs:
    print(x, model(x).data)
