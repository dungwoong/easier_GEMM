# Draft of how to get a custom type
class X:
    def __init__(self):
        self.a = 3

def make_cls(func):
    return type(func.__name__, (X,), {'__module__': func.__module__, '__repr__': func})

@make_cls
def Y(self):
    return str(self.a)

# we could also decorate a class if we wanted but yeah

# Y = type("Y", (X,), {'__repr__': lambda self: str(self.a)})

y = Y()
print(y)