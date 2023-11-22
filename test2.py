
class A():
    def __init__(self):
        pass


def infer_subclasses():
    print([cls.__name__ for cls in A.__subclasses__()])