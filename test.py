class A():
    def __init__(self, x):
        self.x = x

    def check(self):
        self.child = B(self)
        getattr(self.child, 'check')()

    def use(self):
        self.check()
        print(f"a:{self.x}")


class B(A):
    def __init__(self, father):
        self.father = father

    def check(self):
        self.father.x += 1
        print(f"b:{self.father.x}")



if __name__=="__main__":
    # a = B(2,99)
    # a.check()
    # a.see()

    a = A(2)
    a.use()
    # print(a.child)
