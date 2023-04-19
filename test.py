class A:
    def __init__(self, a):
        self.a = a
class B(A):
    def __init__(self, a, na):
        A.__init__(self, a)
        self.a = na
        self.b = 10
        
class C(A):
    def __init__(self, a, c, na):
        A.__init__(self, a)
        self.a = na
        self.c = c

class D(C, B):
    def __init__(self, a, c, na):
        C.__init__(self, a, c, na)
        B.__init__(self, a, na)
        self.a = na
    
a = A(1)
b = B(2, 7)
c = C(3, 4, 8)
d = D(5, 6, 9)

print(a.__dict__)
print(b.__dict__)
print(c.__dict__)
print(d.__dict__)