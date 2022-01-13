import numpy as np
from pybewego.kinematics import Kinematics

class A:
    def __init__(self, a):
        self.a = a

    def get_a(self):
        return self.a

class B(A):
    def get_a(self):
        return self.a


x = B(5)
print(x.get_a())


    