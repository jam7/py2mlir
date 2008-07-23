
class vec(object):
    """
    <4 x float>
    """

    def __init__(self, *args):

        v = []

        if len(args) == 4:
            # vec(1.0, 2.0, 3.0, 4.0)

            for a in args:
                assert isinstance(a, float), ("Arg must be a float type, but %s(%s) is given" % (a, type(a)))
                
                v.append(a)
                
        elif len(args) == 1:
            if isinstance(args[0], list):
                # vec([1.0, 2.0, 3.0, 4.0])
                v = args[0]

            elif isinstance(args[0], float):
                # vec(1.0)
                v.append(args[0])
                v.append(args[0])
                v.append(args[0])
                v.append(args[0])

            else:
                raise Exception("Unsupported input for vec():", args)

        elif len(args) == 0:
            # vec()
            v = [0.0, 0.0, 0.0, 0.0]
           
        else: 
            raise Exception("Unsupported input for vec():", args)

        self.value = v

        # self.addSwizzleMethod()

    def __str__(self):

        return str(self.value)

         
    def __add__(self, b):

        if not isinstance(b, vec):
            raise Exception("RHS is not a type of vec")

        tmp = vec([x + y for x, y in zip(self.value, b.value)])

        return tmp


    def __sub__(self, b):

        if not isinstance(b, vec):
            raise Exception("RHS is not a type of vec")

        tmp = vec([x - y for x, y in zip(self.value, b.value)])

        return tmp


    def __getattr__(self, name):

        #
        # Handle swizzle
        #
        d = { 'x' : 0, 'y' : 1, 'z' : 2, 'w' : 3 }

        assert len(name) < 5, "Invalid attribute: %s" % name

        if len(name) == 1:
            return self.value[d[name]]

        v = vec([0.0, 0.0, 0.0, 0.0])

        for (i, s) in enumerate(name):
            if not d.has_key(s):
                raise Exception("Invalid letter for swizzle:", name)
            
            v.value[i] = self.value[d[s]]

        for i in range(len(name), 4):
            v.value[i] = self.value[d[name[-1]]] 

        return v


"""
TODO...

class mstructdef(object):

    def __init__(self, structElements):

        assert isinstance(structElements, dict)

        print structElements
    

class mstruct(object):

    def __init__(self, structDef):

        assert isinstance(structDef, mstructdef)
        

class marray(object):

    def __init__(self, ty, sz):

        self.type   = ty
        self.size   = sz     
    
        self.values = []

    def __str__(self):

        return str(self.values)
"""

def GetMUDATypeDic():

    d = {
        'vec' : vec
        }

    return d

def printf(msg):

    print msg


def _testVec():

    a = vec([1.0, 2.0, 3.0, 4.0])
    
    print a

    print a.x
    print a.y
    print a.z
    print a.w

    print a.xy
    print a.yz

    b = a.x


def _test():
    import doctest

    _testVec()

    doctest.testmod()

if __name__ == '__main__':
    _test()
