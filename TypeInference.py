import re
import ast

from SymbolTable import *
from MUDA import *

class void(object):
    """
    Represents void type
    """
    def __init__(self):
        pass


class TypeInference(ast.NodeVisitor):
    """
    Simple type inference mechanism for python AST.
    >>> t = TypeInference()
    >>> t.visit(ast.parse("1+3"))
    <type 'int'>
    """

    def __init__(self, symTable):

        assert isinstance(symTable, SymbolTable)

        # First class types
        self.typeDic = {
              'int'    : int
            , 'float'  : float
            , 'None'   : void
            , 'string' : str
            , 'list'   : list
            }

        self.typeDic.update(GetMUDATypeDic())    # register MUDA type

        self.symbolTable = symTable

        # Register intrinsic functions from MUDA module
        self.intrinsics = GetIntrinsicFunctions()

        for (k, v) in self.intrinsics.items():
            retTy  = v[0]
            argTys = v[1]
            sym = Symbol(k, retTy, "function", argtypes = argTys)
            self.symbolTable.append(sym)


    def isFloatType(self, ty):
        if (ty == float or
            ty == vec     ):
            return True

        return False


    def findTypeFromAName(self, name):
        if name in self.typeDic:
            return self.typeDic[name]

        if name == 'void':
            raise Exception("'void' is not a name of python type, please use 'None' instead.", name)

        return None

    def getIntrinsicFunctionFromName(self, name):
        if name in self.intrinsics:
            return self.intrinsics[name]

        return None

    '''
    # Change to use ast.NodeVisitor
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
        # call the method whose name is "infer + ${op_name}"
        #

        method_name = "infer%s" % op_name
        
        if not callable(getattr(self, method_name)):
            raise Exception("Unknown node name:", op_name)

        method = getattr(self, method_name)

        return method(node) 
    '''

    def visit_Module(self, node):

        raise Exception("visit_Module in TypeInference should not be called:", ast.dump(node))
        return self.inferType(node.node)
    

    '''
    # No Stmt in python3 ast
    def visit_Stmt(self, node):

        print("inferType visitStmt", ast.dump(node))
        return self.inferType(node.nodes[0])
    '''

    def checkSwizzleLetter(self, name):

        assert len(name) >= 1 and len(name) < 5

        for s in name:
            if not s in ('x', 'y', 'z', 'w'):
                raise Exception("Not a swizzle letter:", name) 

        return True

    def inferGetattr(self, node):
        """
        a.x
        a.xyz
        a.xyzw
        
        node.expr must be a vector type.
        """

        ty = self.inferType(node.expr)
        assert ty == vec, "swizzle pattern must be specified for vector variable, but variable has type %s: %s" % (ty, node)

        swizzleName = node.attrname
        self.checkSwizzleLetter(swizzleName)

        if len(swizzleName) == 1:
            # scalar
            if ty == vec:
                return float
            else:
                raise Exception("Unknown type:", ty)

        else:
            # vector
            return ty

    '''
    # No Discard in python3 ast
    def inferDiscard(self, node):

        return self.inferType(node.expr)
    '''

    def visit_Call(self, node):

        assert isinstance(node.func, ast.Name)

        print("// => CalFunc:", ast.dump(node))

        # Intrinsic function?
        f = self.getIntrinsicFunctionFromName(node.func.id)
        if f is not None:
            print("// => Intrinsic:", f)
            return f[0]

        # Next, lookup symbol from the symbol table.
        sym = self.symbolTable.find(node.func.id)
        if sym is not None:
            return sym.type

        print("// => not found. func.id=", node.func.id)
        return None

    def visit_UnaryOp(self, node):

        return self.visit(node.operand)

    def visit_BinOp(self, node):
    
        left  = self.visit(node.left)
        right = self.visit(node.right)

        if left != right:
            print("// [type inference] Type mismatch found at line %d: left = %s, right = %s" % (node.lineno, left, right))
            print("//                 node = %s" % ast.dump(node))
            return None

        return left


    #
    # -- Leaf
    #

    '''
    def inferAssName(self, node):

        name = node.name

        # Firstly, name of type?
        ty = self.findTypeFromAName(name)
        if ty is not None:
            return ty

        # Next, lookup symbol
        # return vec
        return None
    '''
    
    def visit_Name(self, node):

        name = node.id

        # Firstly, name of type?
        ty = self.findTypeFromAName(name)
        if ty is not None:
            return ty

        # Next, lookup symbol from the symbol table.
        sym = self.symbolTable.find(name)
        if sym is not None:
            return sym.type

        print("// => not found. name=", name)
        return None


    def visit_Constant(self, node):

        value = node.value

        if value == None:
            return void

        if isinstance(value, type(1.0)):
            return float

        elif isinstance(value, type(1)):
            return int

        elif isinstance(value, type('muda')):
            return str

        else:
            raise Exception("Unknown type of value:", value)
