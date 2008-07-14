import re

from SymbolTable import *
from VecTypes import *

class void(object):
    """
    Represents void type
    """
    def __init__(self):
        pass


class TypeInference(object):
    """
    Simple type inference mechanism for python AST.
    >>> t = TypeInference()
    >>> t.inferType(compiler.parse("1+3"))
    <type 'int'>
    """

    def __init__(self, symTable):

        assert isinstance(symTable, SymbolTable)

        # Intrinsic types
        self.typeDic = {
              'int'   : int
            , 'float' : float
            , 'void'  : void
            }

        self.typeDic.update(GetVecTypeDic())    # register vector type

        self.symbolTable = symTable

    def isTypeName(self, name):
        if self.typeDic.has_key(name):
            return self.typeDic[name]

        return None

    def inferType(self, node):
        """
        Return type if type inference was succeeded, None if failed.
        """

        assert node is not None

        g  = re.compile("(\w+)\(.*\)")
        op = g.match(str(node)) 

        if op == None:
            raise Exception("Invalid node name?", str(node)) 

        op_name = op.group(1)

        #
        # call method whose name is "infer + ${op_name}"
        #

        method_name = "infer%s" % op_name
        
        if not callable(getattr(self, method_name)):
            raise Exception("Unknown node name:", op_name)

        method = getattr(self, method_name)

        return method(node) 

    def inferModule(self, node):

        return self.inferType(node.node)
    

    def inferStmt(self, node):

        return self.inferType(node.nodes[0])


    def inferDiscard(self, node):

        return self.inferType(node.expr)

    def inferCallFunc(self, node):

        print "; => CalFunc:", node
        return self.inferType(node.node)


    def inferAdd(self, node):
    
        left  = self.inferType(node.left)
        right = self.inferType(node.right) 

        if left != right:
            print "; [type inference] Type mismatch found at line %d: left = %s, right = %s" % (node.lineno, left, right)
            print ";                 node = %s" % (node)
            return None

        return left


    #
    # -- Leaf
    #

    def inferAssName(self, node):

        name = node.name

        # Firstly, name of type?
        if self.typeDic.has_key(name):
            return self.typeDic[name]

        # Next, lookup symbol
        # return vec
        return None
    
    def inferName(self, node):

        name = node.name

        # Firstly, name of type?
        if self.typeDic.has_key(name):
            print "; => found type for ", name
            return self.typeDic[name]

        # Next, lookup symbol from the symbol table.
        sym = self.symbolTable.find(name)
        if sym is not None:
            return sym.type

        print "; => not found. name=", name
        return None


    def inferConst(self, node):

        value = node.value

        if value == None:
            return void

        if isinstance(value, type(1.0)):
            return float

        elif isinstance(value, type(1)):
            return int

        else:
            raise Exception("Unknown type of value:", value)
