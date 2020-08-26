import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func 
    
    def backward(self):
        f = self.creator # 1. 関数を取得
        if f is not None:
            x = f.input # 2. 関数のinputを取得
            x.grad = f.backward(self.grad) # 3. 関数のbackwardメソッドを呼ぶ
            x.backward() # 自分より１つ前の変数のbackwardメソッドを呼ぶ（再帰）    

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 出力変数に生みのおやを覚えさせる
        self.input = input
        self.output = output # 出力も覚える
        return output

    def forward(self, x):
        raise NotImplementedError()

    # @staticmethod
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def numerical_diff(f, x, eps=1e-4):
    x0 = (x.data - eps)
    x1 = (x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
# 逆伝播
y.grad = np.array(1.0)
y.backward()
print(x.grad)

print('checking implementation...')
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

print('checking is complete')
print('calculate gradient by myself')
# 逆伝播を試す
y.grad = np.array(1.0)

# --- 一番お尻 ---
C = y.creator # 1. 関数を取得
b = C.input # 2. 関数の入力値を取得
b.grad = C.backward(y.grad) # 3. 関数のbackwordメソッドを呼ぶ
# --- その次 ---
B = b.creator # 1. 関数を取得
a = B.input # 2. 関数の入力値を取得
a.grad = B.backward(b.grad) # 3. 関数のbackwordメソッドを呼ぶ
# --- その次 ---
A = a.creator 
x = A.input
x.grad = A.backward(a.grad)

print(x.grad)