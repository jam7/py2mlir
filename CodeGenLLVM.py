#!/usr/bin/env python

import os, sys
import re
import ast

import llvmlite.ir as ll

# from VecTypes import *
from MUDA import *
from TypeInference import *
from SymbolTable import *


symbolTable    = SymbolTable() 
typer          = TypeInference(symbolTable)

llVoidType     = ll.VoidType()
llIntType      = ll.IntType(32)
llFloatType    = ll.FloatType()
llFVec4Type    = ll.VectorType(llFloatType, 4)
llFVec4PtrType = ll.PointerType(llFVec4Type)
llIVec4Type    = ll.VectorType(llIntType, 4)

def toLLVMTy(ty):

    floatTy = ll.FloatType()

    if ty is None:
        return llVoidType

    d = {
          float : llFloatType
        , int   : llIntType
        , vec   : llFVec4Type
        , void  : llVoidType
        # str   : TODO
        }

    if ty in d:
        return d[ty]

    raise Exception("Unknown type:", ty)
        
        

class DummyIRBuilder:
    """
    Dummy IR Builder to ignore all requests.

    py2llvm performs type inference and decide the type of return value.
    In order to do that, py2llvm requires two passes.  1st pass to infer
    return type.  2nd pass to generate code.  This dummy IR builder is
    used in this 1st pass.
    """

    def ret_void(self):
        return 'dummyIR'
    def ret(self, *args, **kwargs):
        return 'dummyIR'
    def alloca(self, *args, **kwargs):
        return 'dummyIR'
    def load(self, *args, **kwargs):
        return 'dummyIR'
    def store(self, *args, **kwargs):
        return 'dummyIR'
    def add(self, *args, **kwargs):
        return 'dummyIR'
    def sub(self, *args, **kwargs):
        return 'dummyIR'
    def mul(self, *args, **kwargs):
        return 'dummyIR'
    def fdiv(self, *args, **kwargs):
        return 'dummyIR'
    def call(self, *args, **kwargs):
        return 'dummyIR'
    def fcmp(self, *args, **kwargs):
        return 'dummyIR'
    def sext(self, *args, **kwargs):
        return 'dummyIR'
    def insert_element(self, *args, **kwargs):
        return 'dummyIR'
    def extract_element(self, *args, **kwargs):
        return 'dummyIR'

class CodeGenLLVM(ast.NodeVisitor):
    """
    LLVM CodeGen class 
    """

    def __init__(self):

        self.body             = ""
        self.globalscope      = ""

        self.module           = ll.Module("module")
        self.funcs            = []
        self.func             = None # Current function
        self.builder          = None

        self.currFuncRetType  = None
        self.prevFuncRetNode  = None    # for reporiting err

        self.externals        = {}

    def visit_Module(self, node):

        # emitExternalSymbols() should be called before self.visit(node.node)
        self.emitExternalSymbols()

        for stmt in node.body:
            if isinstance(stmt, ast.AST):
                self.visit(stmt)

        print(self.module)  # Output LLVM code to stdout.
        print(self.emitCommonHeader())


    '''
    # No Print and Printnl in python3 ast
    def visit_Print(self, node):
        return None # Discard

    def visitPrintnl(self, node):
        return None # Discard
    '''


    def visit_Return(self, node):

        ty   = typer.visit(node.value)
        print("; Return ty = ", ty)

        # Return(Const(None))
        if isinstance(node.value, ast.NameConstant):
            if node.value.value == None:
                self.currFuncRetType = void
                self.prevFuncRetNode = node
                return self.builder.ret_void()
                
        expr = self.visit(node.value)

        if self.currFuncRetType is None:
            self.currFuncRetType = ty
            self.prevFuncRetNode = node

        elif self.currFuncRetType != ty:
            raise Exception("Different type for return expression: expected %s(lineno=%d, %s) but got %s(lineno=%d, %s)" % (self.currFuncRetType, self.prevFuncRetNode.lineno, self.prevFuncRetNode, ty, node.lineno, ast.dump(node)))

        return self.builder.ret(expr)

    def mkFunctionSignature(self, retTy, node):

        # Argument should have default value which represents type of argument.
        if len(node.args.args) != len(node.args.defaults):
            raise Exception("Function argument should have default values which represents type of the argument:", node)

        argLLTys = []

        for (name, tyname) in zip(node.args.args, node.args.defaults):

            assert isinstance(tyname, ast.Name)

            ty = typer.isNameOfFirstClassType(tyname.id)
            if ty is None:
                raise Exception("Unknown name of type:", tyname.id)

            llTy = toLLVMTy(ty)
            
            # vector argument is passed by pointer.
            # if llTy == llFVec4Type:
            #     llTy = llFVec4PtrType

            argLLTys.append(llTy)

        funcLLVMTy = ll.FunctionType(retTy, argLLTys)
        func = ll.Function(self.module, funcLLVMTy, node.name)

        # Assign name for each arg
        for i, name in enumerate(node.args.args):

            # if llTy == llFVec4Type:
            #     argname = name + "_p"
            # else: 
            #     argname = name
            argname = name.arg

            func.args[i].name = argname


        return func
        
    def visit_FunctionDef(self, node):

        """
        Perform 2 passes to translate python function ot LLVM IR.
        1st pass parse python function to infer a return type.
        2nd pass parse python function and translate it to LLVM IR.

        Original implementation was translating python twice and
        discarding first result.  However, llvmlite doesn't have
        such discarding mechanism, so I create dummyIRBuilder to
        ignore all requests and use it in 1st pass.
        """

        # init
        self.currFuncRetType = None 
        

        symbolTable.pushScope(node.name)
        retLLVMTy    = llVoidType # Dummy
        builder      = DummyIRBuilder()
        self.builder = builder

        # Add function argument to symblol table.
        # And emit function prologue.
        for i, (name, tyname) in enumerate(zip(node.args.args, node.args.defaults)):

            ty = typer.isNameOfFirstClassType(tyname.id)

            bufSym = symbolTable.genUniqueSymbol(ty)

            symbolTable.append(Symbol(name.arg, ty, "variable", llstorage=None))

        for stmt in node.body:
            if isinstance(stmt, ast.AST):
                self.visit(stmt)
        symbolTable.popScope()

        symbolTable.pushScope(node.name)
        retLLVMTy    = toLLVMTy(self.currFuncRetType)
        func         = self.mkFunctionSignature(retLLVMTy, node)
        entry        = func.append_basic_block("entry")
        builder      = ll.IRBuilder(entry)
        self.func    = func
        self.builder = builder
        self.funcs.append(func)

        # Add function argument to symblol table.
        # And emit function prologue.
        for i, (name, tyname) in enumerate(zip(node.args.args, node.args.defaults)):

            ty = typer.isNameOfFirstClassType(tyname.id)

            bufSym = symbolTable.genUniqueSymbol(ty)

            llTy = toLLVMTy(ty)

            # if llTy == llFVec4Type:
            #     # %name.buf = alloca ty
            #     # %tmp = load %arg
            #     # store %tmp, %name.buf
            #     allocaInst = self.builder.alloca(llTy, bufSym.name)
            #     pTy = llFVec4PtrType
            #     tmpSym     = symbolTable.genUniqueSymbol(ty)
            #     loadInst   = self.builder.load(func.args[i], tmpSym.name)
            #     storeInst  = self.builder.store(loadInst, allocaInst)
            #     symbolTable.append(Symbol(name, ty, "variable", llstorage=allocaInst))
            # else:
            #     # %name.buf = alloca ty
            #     # store val, %name.buf
            #     allocaInst = self.builder.alloca(llTy, bufSym.name)
            #     storeInst  = self.builder.store(func.args[i], allocaInst)
            #     symbolTable.append(Symbol(name, ty, "variable", llstorage=allocaInst))
            # %name.buf = alloca ty
            # store val, %name.buf
            allocaInst = self.builder.alloca(llTy, name=bufSym.name)
            storeInst  = self.builder.store(func.args[i], allocaInst)
            symbolTable.append(Symbol(name.arg, ty, "variable", llstorage=allocaInst))

        for stmt in node.body:
            if isinstance(stmt, ast.AST):
                self.visit(stmt)

        if self.currFuncRetType is None:
            # Add ret void.
            self.builder.ret_void()
            self.currFuncRetType = void

        symbolTable.popScope()

        # Register function to symbol table
        symbolTable.append(Symbol(node.name, self.currFuncRetType, "function", llstorage=func))



    '''
    # No Stmt in python3 ast
    def visit_Stmt(self, node):

        for node in node.nodes:

            print("; [stmt]", node)

            self.visit(node)
    '''

    def visit_Assign(self, node):

        if len(node.targets) != 1:
            raise Exception("TODO:", ast.dump(node))

        print("; [Asgn]")
        rTy     = typer.visit(node.value)
        print("; [Asgn]. rTy = ", rTy)

        print("; [Asgn]. node.value = ", node.value)
        rLLInst = self.visit(node.value)
        print("; [Asgn]. rhs = ", rLLInst)

        lhsNode = node.targets[0]

        lTy = None
        if isinstance(lhsNode, ast.Name):

            sym = symbolTable.find(lhsNode.id)
            if sym is None:
                # The variable appears here firstly.

                # alloc storage
                llTy = toLLVMTy(rTy)
                llStorage = self.builder.alloca(llTy, name=lhsNode.id)

                sym = Symbol(lhsNode.id, rTy, "variable", llstorage = llStorage)
                symbolTable.append(sym)
                print("; [Sym] New symbol added: ", sym)

                lTy = rTy

            else:
                # symbol is already defined.
                lTy = sym.type



        if rTy != lTy:
            raise Exception("ERR: TypeMismatch: lTy = %s, rTy = %s: %s" % (lTy, rTy, ast.dump(node)))

        lSym = symbolTable.find(lhsNode.id)

        storeInst = self.builder.store(rLLInst, lSym.llstorage)

        print(";", storeInst)

        print("; [Asgn]", ast.dump(node))
        print("; [Asgn] nodes = ", node.targets)
        print("; [Asgn] expr  = ", node.value)

        # No return

    '''
    def visitIf(self, node):

        print("; ", node.tests)
        print("; ", node.else_)

        raise Exception("muda")

    def emitVCompare(self, op, lInst, rInst):

        d = { "==" : llvm.core.RPRED_OEQ
            , "!=" : llvm.core.RPRED_ONE
            , ">"  : llvm.core.RPRED_OGT 
            , ">=" : llvm.core.RPRED_OGE
            , "<"  : llvm.core.RPRED_OLT 
            , "<=" : llvm.core.RPRED_OLE
            }

        llop = d[op]

        i0 = llvm.core.Constant.int(llIntType, 0);
        i1 = llvm.core.Constant.int(llIntType, 1);
        i2 = llvm.core.Constant.int(llIntType, 2);
        i3 = llvm.core.Constant.int(llIntType, 3);
        vizero = llvm.core.Constant.vector([llvm.core.Constant.int(llIntType, 0)] * 4)

        tmp0  = symbolTable.genUniqueSymbol(float)
        tmp1  = symbolTable.genUniqueSymbol(float)
        tmp2  = symbolTable.genUniqueSymbol(float)
        tmp3  = symbolTable.genUniqueSymbol(float)
        tmp4  = symbolTable.genUniqueSymbol(float)
        tmp5  = symbolTable.genUniqueSymbol(float)
        tmp6  = symbolTable.genUniqueSymbol(float)
        tmp7  = symbolTable.genUniqueSymbol(float)
        le0   = self.builder.extract_element(lInst, i0, tmp0.name) 
        le1   = self.builder.extract_element(lInst, i1, tmp1.name) 
        le2   = self.builder.extract_element(lInst, i2, tmp2.name) 
        le3   = self.builder.extract_element(lInst, i3, tmp3.name) 
        re0   = self.builder.extract_element(rInst, i0, tmp4.name) 
        re1   = self.builder.extract_element(rInst, i1, tmp5.name) 
        re2   = self.builder.extract_element(rInst, i2, tmp6.name) 
        re3   = self.builder.extract_element(rInst, i3, tmp7.name) 

        ftmp0 = symbolTable.genUniqueSymbol(float)
        ftmp1 = symbolTable.genUniqueSymbol(float)
        ftmp2 = symbolTable.genUniqueSymbol(float)
        ftmp3 = symbolTable.genUniqueSymbol(float)

        f0 = self.builder.fcmp(llop, le0, re0, ftmp0.name)
        f1 = self.builder.fcmp(llop, le1, re1, ftmp1.name)
        f2 = self.builder.fcmp(llop, le2, re2, ftmp2.name)
        f3 = self.builder.fcmp(llop, le3, re3, ftmp3.name)

        # i1 -> i32
        ctmp0 = symbolTable.genUniqueSymbol(int)
        ctmp1 = symbolTable.genUniqueSymbol(int)
        ctmp2 = symbolTable.genUniqueSymbol(int)
        ctmp3 = symbolTable.genUniqueSymbol(int)
        c0 = self.builder.sext(f0, llIntType)
        c1 = self.builder.sext(f1, llIntType)
        c2 = self.builder.sext(f2, llIntType)
        c3 = self.builder.sext(f3, llIntType)

        # pack
        s0 = symbolTable.genUniqueSymbol(llIVec4Type)
        s1 = symbolTable.genUniqueSymbol(llIVec4Type)
        s2 = symbolTable.genUniqueSymbol(llIVec4Type)
        s3 = symbolTable.genUniqueSymbol(llIVec4Type)

        r0 = self.builder.insert_element(vizero, c0, i0, s0.name)
        r1 = self.builder.insert_element(r0    , c1, i1, s1.name)
        r2 = self.builder.insert_element(r1    , c2, i2, s2.name)
        r3 = self.builder.insert_element(r2    , c3, i3, s3.name)

        return r3

    def visitCompare(self, node):

        print("; ", node.expr)
        print("; ", node.ops[0])

        lTy = typer.inferType(node.expr)
        rTy = typer.inferType(node.ops[0][1])

        if rTy != lTy:
            raise Exception("ERR: TypeMismatch: lTy = %s, rTy = %s for %s, line %d" % (lTy, rTy, node, node.lineno))

        lLLInst = self.visit(node.expr)
        rLLInst = self.visit(node.ops[0][1])

        op  = node.ops[0][0]

        if rTy == vec:
            return self.emitVCompare(op, lLLInst, rLLInst)

        if op == "<":
            print("muda")
        elif op == ">":
            print("muda")
        else:
            raise Exception("Unknown operator:", op)

        raise Exception("muda")
    '''

    def visit_UnaryOp(self, node):

        assert isinstance(node.op, ast.USub)
        ty       = typer.visit(node.operand)
        e        = self.visit(node.operand)
        zeroInst = ll.Constant(toLLVMTy(ty), 0)
        tmpSym   = symbolTable.genUniqueSymbol(ty)

        subInst = self.builder.sub(zeroInst, e, tmpSym.name)

        return subInst

    '''
    def visitGetattr(self, node):

        d = { 'x' : llvm.core.Constant.int(llIntType, 0)
            , 'y' : llvm.core.Constant.int(llIntType, 1)
            , 'z' : llvm.core.Constant.int(llIntType, 2)
            , 'w' : llvm.core.Constant.int(llIntType, 3)
            }


        ty = typer.inferType(node)
        print("; getattr: expr", node.expr)
        print("; getattr: attrname", node.attrname)
        print("; getattr: ty", ty)

        rLLInst  = self.visit(node.expr)
        tmpSym   = symbolTable.genUniqueSymbol(ty)

        if len(node.attrname) == 1:
            # emit extract element
            s = node.attrname[0]

            inst = self.builder.extract_element(rLLInst, d[s], tmpSym.name)

        return inst
        '''
        

    def visit_BinOp(self, node):

        lTy = typer.visit(node.left)
        rTy = typer.visit(node.right)

        if rTy != lTy:
            raise Exception("ERR: TypeMismatch: lTy = %s, rTy = %s for %s, line %d" % (lTy, rTy, ast.dump(node), node.lineno))

        lLLInst = self.visit(node.left)
        rLLInst = self.visit(node.right)
        
        tmpSym = symbolTable.genUniqueSymbol(lTy)

        if isinstance(node.op, ast.Add):
            inst = self.builder.add(lLLInst, rLLInst, tmpSym.name)
            print("; [AddOp] inst = ", inst)
        elif isinstance(node.op, ast.Sub):
            inst = self.builder.sub(lLLInst, rLLInst, tmpSym.name)
            print("; [SubOp] inst = ", inst)
        elif isinstance(node.op, ast.Mul):
            inst = self.builder.mul(lLLInst, rLLInst, tmpSym.name)
            print("; [MulOp] inst = ", inst)
        elif isinstance(node.op, ast.Div):
            if typer.isFloatType(lTy):
                inst = self.builder.fdiv(lLLInst, rLLInst, tmpSym.name)
            else:
                raise Exception("TODO: div for type: ", lTy)

            print("; [DIvOp] inst = ", inst)

        return inst

    '''
    def handleInitializeTypeCall(self, ty, args):

        llty = toLLVMTy(ty)

        if llty == llFVec4Type:

            i0 = llvm.core.Constant.int(llIntType, 0);
            i1 = llvm.core.Constant.int(llIntType, 1);
            i2 = llvm.core.Constant.int(llIntType, 2);
            i3 = llvm.core.Constant.int(llIntType, 3);

            vf = llvm.core.Constant.vector([llvm.core.Constant.real(llFloatType, "0.0")] * 4)

            # args =  [List([float, float, float, float])]
            #      or [List(float)]
            
            if isinstance(args[0], list):
                elems = args[0]
            else:
                elems = [args[0], args[0], args[0], args[0]]

            s0 = symbolTable.genUniqueSymbol(llFVec4Type)
            s1 = symbolTable.genUniqueSymbol(llFVec4Type)
            s2 = symbolTable.genUniqueSymbol(llFVec4Type)
            s3 = symbolTable.genUniqueSymbol(llFVec4Type)

            r0 = self.builder.insert_element(vf, elems[0] , i0, s0.name)
            r1 = self.builder.insert_element(r0, elems[1] , i1, s1.name)
            r2 = self.builder.insert_element(r1, elems[2] , i2, s2.name)
            r3 = self.builder.insert_element(r2, elems[3] , i3, s3.name)

            return r3
        
    def emitVSel(self, node):

        self.builder.call
        f3 = self.builder.call(func, [e3], ftmp3.name)
    '''
        
    def visit_Call(self, node):

        assert isinstance(node.func, ast.Name)

        print("; callfunc", ast.dump(node.args))

        args = [self.visit(a) for a in node.args]

        print("; callfuncafter", args)

        print("; Call ", ast.dump(node))
        print("; Call func ", node.func.id)

        '''

        print("; callfunc", node.args)

        ty = typer.isNameOfFirstClassType(node.node.name)
        print("; callfuncafter: ty = ", ty)

        #
        # value initialier? 
        #
        if ty:
            # int, float, vec, ...
            return self.handleInitializeTypeCall(ty, args)
                
        
        #
        # vector math function?
        # 
        ret = self.isVectorMathFunction(node.node.name)
        if ret is not False:
            return self.emitVMath(ret[1], args)

        #
        # Special function?
        #
        if (node.node.name == "vsel"):
            func = self.getExternalSymbolInstruction("vsel")
            tmp  = symbolTable.genUniqueSymbol(vec)

            print("; ", args)
            c    = self.builder.call(func, args, tmp.name)

            return c
            
        #
        # Defined in the source?
        #
        ty      = typer.inferType(node.node)
        funcSig = symbolTable.lookup(node.node.name)

        if funcSig.kind is not "function":
            raise Exception("Symbol isn't registered as function:", node.node.name)

        # emit call 
        tmp  = symbolTable.genUniqueSymbol(vec)
        return self.builder.call(funcSig.llstorage, args, tmp.name)
        '''


    def visit_List(self, node):

        return [self.visit(a) for a in node.elts]

    #
    # Leaf
    #
    def visit_Name(self, node):

        sym = symbolTable.lookup(node.id)

        tmpSym = symbolTable.genUniqueSymbol(sym.type)

        # %tmp = load %name

        loadInst = self.builder.load(sym.llstorage, tmpSym.name)

        print("; [Leaf] inst = ", loadInst)
        return loadInst

    '''
    # No Discard in python3 ast
    def visitDiscard(self, node):

        self.visit(node.expr)

        #
        # return None
        #
    '''


    def mkLLConstInst(self, ty, value):

        # ty = typer.inferType(node)
        # print("; [Typer] %s => %s" % (str(node), str(ty)))

        llTy   = toLLVMTy(ty)
        bufSym = symbolTable.genUniqueSymbol(ty)
        tmpSym = symbolTable.genUniqueSymbol(ty)

        # %tmp  = alloca ty
        # store ty val, %tmp
        # %inst = load ty, %tmp

        allocInst = self.builder.alloca(llTy, name=bufSym.name)

        llConst   = None
        if llTy   == llIntType:
            llConst = ll.Constant(llIntType, value)
    
        elif llTy == llFloatType:
            llConst = ll.Constant(llFloatType, value)

        elif llTy == llFVec4Type:
            print(";", value)
            raise Exception("muda")
    
        storeInst = self.builder.store(llConst, allocInst)
        loadInst  = self.builder.load(allocInst, tmpSym.name)

        print(";", loadInst)

        return loadInst

    def visit_Constant(self, node):
        
        ty = typer.visit(node)

        return self.mkLLConstInst(ty, node.value)

    def emitCommonHeader(self):

        s = """
define <4 x float> @vsel(<4 x float> %a, <4 x float> %b, <4 x i32> %mask) {
entry:
    %a.i     = bitcast <4 x float> %a to <4 x i32>
    %b.i     = bitcast <4 x float> %b to <4 x i32>
    %tmp0    = and <4 x i32> %b.i, %mask
    %tmp.addr = alloca <4 x i32>
    store <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32>* %tmp.addr
    %allone  = load <4 x i32>, <4 x i32>* %tmp.addr
    %invmask = xor <4 x i32> %allone, %mask
    %tmp1    = and <4 x i32> %a.i, %invmask
    %tmp2    = or <4 x i32> %tmp0, %tmp1
    %r       = bitcast <4 x i32> %tmp2 to <4 x float>

    ret <4 x float> %r
}

"""
        return s

    #
    #
    #
    def emitExternalSymbols(self):

        d = {
              'fabsf'  : ( llFloatType, [llFloatType] )
            , 'expf'   : ( llFloatType, [llFloatType] )
            , 'logf'   : ( llFloatType, [llFloatType] )
            , 'sqrtf'  : ( llFloatType, [llFloatType] )
            # Don't declare vsel since it causes duplicated definition error
            # , 'vsel'   : ( llFVec4Type, [llFVec4Type, llFVec4Type, llIVec4Type] )
            }

        for k, v in d.items():
            fty = ll.FunctionType(v[0], v[1])
            f   = ll.Function(self.module, fty, k)

            self.externals[k] = f

    '''
    def getExternalSymbolInstruction(self, name):

        if name in self.externals:
            return self.externals[name]
        else:
            raise Exception("Unknown external symbol:", name, self.externals)

    def isExternalSymbol(self, name):
        if name in self.externals:
            return True
        else:
            return False

    #
    # Vector math
    #
    def isVectorMathFunction(self, name):
        d = {
              'vabs'  : 'fabsf'
            , 'vexp'  : 'expf'
            , 'vlog'  : 'logf'
            , 'vsqrt' : 'sqrtf'
            }

        if name in d:
            return (True, d[name])
        else:
            return False

    def emitVMath(self, fname, llargs):
        """
        TODO: Use MUDA's optimized vector math function for LLVM.
        """

        i0    = llvm.core.Constant.int(llIntType, 0)
        i1    = llvm.core.Constant.int(llIntType, 1)
        i2    = llvm.core.Constant.int(llIntType, 2)
        i3    = llvm.core.Constant.int(llIntType, 3)
        vzero = llvm.core.Constant.vector([llvm.core.Constant.real(llFloatType, "0.0")] * 4)

        func = self.getExternalSymbolInstruction(fname)

        # Decompose vector element
        tmp0  = symbolTable.genUniqueSymbol(float)
        tmp1  = symbolTable.genUniqueSymbol(float)
        tmp2  = symbolTable.genUniqueSymbol(float)
        tmp3  = symbolTable.genUniqueSymbol(float)
        e0    = self.builder.extract_element(llargs[0], i0, tmp0.name) 
        e1    = self.builder.extract_element(llargs[0], i1, tmp1.name) 
        e2    = self.builder.extract_element(llargs[0], i2, tmp2.name) 
        e3    = self.builder.extract_element(llargs[0], i3, tmp3.name) 

        ftmp0 = symbolTable.genUniqueSymbol(float)
        ftmp1 = symbolTable.genUniqueSymbol(float)
        ftmp2 = symbolTable.genUniqueSymbol(float)
        ftmp3 = symbolTable.genUniqueSymbol(float)
        f0 = self.builder.call(func, [e0], ftmp0.name)
        f1 = self.builder.call(func, [e1], ftmp1.name)
        f2 = self.builder.call(func, [e2], ftmp2.name)
        f3 = self.builder.call(func, [e3], ftmp3.name)

        # pack
        s0 = symbolTable.genUniqueSymbol(llFVec4Type)
        s1 = symbolTable.genUniqueSymbol(llFVec4Type)
        s2 = symbolTable.genUniqueSymbol(llFVec4Type)
        s3 = symbolTable.genUniqueSymbol(llFVec4Type)

        r0 = self.builder.insert_element(vzero, f0, i0, s0.name)
        r1 = self.builder.insert_element(r0   , f1, i1, s1.name)
        r2 = self.builder.insert_element(r1   , f2, i2, s2.name)
        r3 = self.builder.insert_element(r2   , f3, i3, s3.name)
        return r3

        # r0 = self.builder.insert_element(vzero, f2, i2, s0.name)
        # r1 = self.builder.insert_element(r0   , e1, i1, s1.name)
        # r2 = self.builder.insert_element(r1   , e2, i0, s2.name)
        # return r2

    def emitVAbs(self, llargs):

        return self.emitVMath("fabsf", llargs)
    '''
        

def _test():
    import doctest
    doctest.testmod()
    sys.exit()
    

def main():

    if len(sys.argv) < 2:
        _test()

    r = open(sys.argv[1], 'r')
    mod = ast.parse(r.read())
    # print(ast.dump(mod))

    codegen = CodeGenLLVM()
    codegen.visit(mod)

if __name__ == '__main__':
    main()
