
from test2 import A, infer_subclasses
from models.lstm import LSTM

from typing import Type


class B(A):
    def __init__(self):
        pass


class C(A):
    def __init__(self):
        pass


def classname(cls):
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name


class DemoMetaClass(type): 
    def __new__(cls: Type['DemoMetaClass'], name: str, bases: tuple, defcl: dict) -> 'DemoMetaClass':
        obj = super().__new__(cls, name, bases, defcl)
        obj.MODEL_PATH = None if obj.MODEL is None else classname(obj.MODEL)
        return obj


class SupTest(metaclass=DemoMetaClass):
    MODEL = None


class Test(SupTest):
    MODEL = LSTM

if __name__ == '__main__':
    print(Test.MODEL_PATH)
    # infer_subclasses()
    # t = Test()
    # print(t.MODEL_PATH)
