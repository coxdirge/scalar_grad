from engine import Value

# a = Value(2.0)
# b = Value(-3.0)
#
# c = a * b
# d = c + a
#
# print(c)
# print(d)
#
# assert c.data == -6.0
# assert d.data == -4.0
#
# assert c._prev == (a, b)
# assert d._prev == (c, a)
#
# c.grad = 1.0
# c._backward()
#
# print(a.grad)
# print(b.grad)

a = Value(2.0)
b = Value(-3.0)

c = a * b
d = c + a

d.backward()

print(a.grad)  # -2.0
print(b.grad)  #  2.0
print(c.grad)  #  1.0
print(d.grad)  #  1.0

a = Value(2.0)
b = Value(4.0)

y = (a - b) / b + a ** 3

y.backward()

print(y.data)  # 7.5
print(a.grad)  # 12.25
print(b.grad)  # -0.125

x1 = Value(2.0)
x2 = Value(0.0)

w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.881373587)

z = x1 * w1 + x2 * w2 + b
y = z.tanh()

y.backward()

print(y.data)
print(w1.grad)
print(w2.grad)
print(b.grad)
