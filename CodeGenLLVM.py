#!/usr/bin/env python

import os, sys
import re
import compiler

import llvm.core

from VecTypes import *
from TypeInference import *
from SymbolTable import *


symbolTable = SymbolTable() 
typer       = TypeInference(symbolTable)

llVoidType    = llvm.core.Type.void()
llIntType     = llvm.core.Type.int()
llFloatType   = llvm.core.Type.float()
llFVec4Type   = llvm.core.Type.vector(llFloatType, 4)

def toLLVMTy(ty):

    floatTy = llvm.core.Type.float()

    if ty is None:
        return llVoidType

    d = {
          float : llFloatType
        , int   : llIntType
        , vec   : llFVec4Type
        , void  : llVoidType
        }

    if d.has_key(ty):
        return d[ty]

    raise Exception("Unknown type:", ty)
        
        

class CodeGenLLVM:
    """
    LLVM CodeGen class 
    """

    def __init__(self):

        self.body             = ""
        self.globalscope      = ""

        self.module           = llvm.core.Module.new("module")
        self.funcs            = []
        self.func             = None # Current function
        self.builder          = None

        self.currFuncRetType  = None
        self.prevFuncRetNode  = None    # for reporiting err

    def visitReturn(self, node):

        ty   = typer.inferType(node.value)
        print "; Return ty = ", ty

        # Return(Const(None))
        if isinstance(node.value, compiler.ast.Const):
            if node.value.value == None:
                self.currFuncRetType = void
                self.prevFuncRetNode = node
                return self.builder.ret_void()
                
        expr = self.visit(node.value)

        if self.currFuncRetType is None:
            self.currFuncRetType = ty
            self.prevFuncRetNode = node

        elif self.currFuncRetType != ty:
            raise Exception("Different type for return expression: expected %s(lineno=%d, %s) but got %s(lineno=%d, %s)" % (self.currFuncRetType, self.prevFuncRetNode.lineno, self.prevFuncRetNode, ty, node.lineno, node))

        return self.builder.ret(expr)

    def mkFunctionSignature(self, retTy, node):

        # Argument should have default value which represents type of argument.
        if len(node.argnames) != len(node.defaults):
            raise Exception("Function argument should have default values which represents type of the argument:", node)

        argLLTys = []

        for (name, tyname) in zip(node.argnames, node.defaults):

            assert isinstance(tyname, compiler.ast.Name)

            ty = typer.isTypeName(tyname.name)
            if ty is None:
                raise Exception("Unknown name of type:", tyname.name)

            argLLTys.append(toLLVMTy(ty))

        # TODO: Infer return type.
        funcLLVMTy = llvm.core.Type.function(retTy, argLLTys)
        func = llvm.core.Function.new(self.module, funcLLVMTy, node.name)

        # Assign name for each arg
        for i, name in enumerate(node.argnames):
            func.args[i].name = name


        return func
        
    def visitFunction(self, node):

        """
        Do nasty trick to handle return type of function correctly.

        We visit node AST two times.
        First pass just determines return type of the function
        (All LLVM code body generated are discarded).
        Then second pass we emit LLVM code body with return type found
        in the first pass.
        """


        symbolTable.pushScope(node.name)
        retLLVMTy    = llvm.core.Type.void() # Dummy
        func         = self.mkFunctionSignature(retLLVMTy, node)
        entry        = func.append_basic_block("entry")
        builder      = llvm.core.Builder.new(entry)
        self.func    = func
        self.builder = builder
        self.funcs.append(func)

        # Add function argument to symblol table.
        # And emit function prologue.
        for i, (name, tyname) in enumerate(zip(node.argnames, node.defaults)):

            ty = typer.isTypeName(tyname.name)

            # %name.buf = alloca ty
            # store val, %name.buf
            bufSym = symbolTable.genUniqueSymbol(ty)
            allocaInst = self.builder.alloca(toLLVMTy(ty), bufSym.name)
            storeInst  = self.builder.store(func.args[i], allocaInst)
            symbolTable.append(Symbol(name, ty, llstorage=allocaInst))

        self.visit(node.code)
        symbolTable.popScope()

        # Discard llvm code except for return type 
        func.delete()
        del(self.funcs[-1])


        symbolTable.pushScope(node.name)
        retLLVMTy    = toLLVMTy(self.currFuncRetType)
        func         = self.mkFunctionSignature(retLLVMTy, node)
        entry        = func.append_basic_block("entry")
        builder      = llvm.core.Builder.new(entry)
        self.func    = func
        self.builder = builder
        self.funcs.append(func)

        # Add function argument to symblol table.
        # And emit function prologue.
        for i, (name, tyname) in enumerate(zip(node.argnames, node.defaults)):

            ty = typer.isTypeName(tyname.name)

            # %name.buf = alloca ty
            # store val, %name.buf
            bufSym = symbolTable.genUniqueSymbol(ty)
            allocaInst = self.builder.alloca(toLLVMTy(ty), bufSym.name)
            storeInst  = self.builder.store(func.args[i], allocaInst)
            symbolTable.append(Symbol(name, ty, llstorage=allocaInst))

        self.visit(node.code)
        symbolTable.popScope()

        print self.module   # Output LLVM code to stdout.


    def visitStmt(self, node):

        for node in node.nodes:

            print "; [stmt]", node  

            self.visit(node)

    def visitAssign(self, node):

        if len(node.nodes) != 1:
            raise Exception("TODO:", node)

        print "; [Asgn]"
        rTy     = typer.inferType(node.expr)
        print "; [Asgn]. rTy = ", rTy

        print "; [Asgn]. node.expr = ", node.expr
        rLLInst = self.visit(node.expr)
        print "; [Asgn]. rhs = ", rLLInst

        lhsNode = node.nodes[0]

        lTy = None
        if isinstance(lhsNode, compiler.ast.AssName):

            sym = symbolTable.find(lhsNode.name)
            if sym is None:
                # The variable appears here firstly.

                # alloc storage
                llTy = toLLVMTy(rTy)
                llStorage = self.builder.alloca(llTy, lhsNode.name) 

                sym = Symbol(lhsNode.name, rTy, llstorage = llStorage)
                symbolTable.append(sym)
                print "; [Sym] New symbol added: ", sym

                lTy = rTy



        if rTy != lTy:
            raise Exception("ERR: TypeMismatch: lTy = %s, rTy = %s: %s" % (lTy, rTy, node)) 

        lSym = symbolTable.find(lhsNode.name)

        storeInst = self.builder.store(rLLInst, lSym.llstorage)

        print ";", storeInst

        print "; [Asgn]", node  
        print "; [Asgn] nodes = ", node.nodes 
        print "; [Asgn] expr  = ", node.expr

        # No return


    def visitAdd(self, node):

        lTy = typer.inferType(node.left)
        rTy = typer.inferType(node.right)

        if rTy != lTy:
            raise Exception("ERR: TypeMismatch: lTy = %s, rTy = %s for %s, line %d" % (lTy, rTy, node, node.lineno))

        lLLInst = self.visit(node.left)
        rLLInst = self.visit(node.right)
        
        tmpSym = symbolTable.genUniqueSymbol(lTy)

        addInst = self.builder.add(lLLInst, rLLInst, tmpSym.name)
        print "; [AddOp] inst = ", addInst

        return addInst


    def handleInitializeTypeCall(self, ty, args):

        llty = toLLVMTy(ty)

        if llty == llFVec4Type:

            i0 = llvm.core.Constant.int(llIntType, 0);
            i1 = llvm.core.Constant.int(llIntType, 1);
            i2 = llvm.core.Constant.int(llIntType, 2);
            i3 = llvm.core.Constant.int(llIntType, 3);

            vf = llvm.core.Constant.vector([llvm.core.Constant.real(llFloatType, "0.0")] * 4)

            # args = [List([float, float, float, float])]
            elems = args[0]

            s0 = symbolTable.genUniqueSymbol(llFVec4Type)
            s1 = symbolTable.genUniqueSymbol(llFVec4Type)
            s2 = symbolTable.genUniqueSymbol(llFVec4Type)
            s3 = symbolTable.genUniqueSymbol(llFVec4Type)

            r0 = self.builder.insert_element(vf, elems[0] , i0, s0.name)
            r1 = self.builder.insert_element(r0, elems[1] , i1, s1.name)
            r2 = self.builder.insert_element(r1, elems[2] , i2, s2.name)
            r3 = self.builder.insert_element(r2, elems[3] , i3, s3.name)

            return r3
        
        
    def visitCallFunc(self, node):

        assert isinstance(node.node, compiler.ast.Name)

        print "; callfunc", node.args

        args = [self.visit(a) for a in node.args]

        print "; callfuncafter", args

        ty = typer.isTypeName(node.node.name)
        print "; callfuncafter: ty = ",ty
        if ty:
            # int, float, vec, ...
            return self.handleInitializeTypeCall( ty, args )
                

        ty = typer.inferType(node.node)

        raise Exception("TODO...")
        
    def visitList(self, node):

        return [self.visit(a) for a in node.nodes]

    #
    # Leaf
    #
    def visitName(self, node):

        sym = symbolTable.lookup(node.name) 

        tmpSym = symbolTable.genUniqueSymbol(sym.type)

        # %tmp = load %name

        loadInst = self.builder.load(sym.llstorage, tmpSym.name)

        print "; [Leaf] inst = ", loadInst
        return loadInst


    def visitConst(self, node):

        ty = typer.inferType(node)
        print "; [Typer] %s => %s" % (str(node), str(ty))

        llTy   = toLLVMTy(ty)
        bufSym = symbolTable.genUniqueSymbol(ty)
        tmpSym = symbolTable.genUniqueSymbol(ty)

        # %tmp  = alloca ty
        # store ty val, %tmp
        # %inst = load ty, %tmp

        allocInst = self.builder.alloca(llTy, bufSym.name)

        llConst   = None
        if llTy   == llIntType:
            llConst = llvm.core.Constant.int(llIntType, node.value)
    
        elif llTy == llFloatType:
            llConst = llvm.core.Constant.real(llFloatType, node.value)

        elif llTy == llFVec4Type:
            print ";", node.value
            raise Exception("muda")
    
        storeInst = self.builder.store(llConst, allocInst)
        loadInst  = self.builder.load(allocInst, tmpSym.name)

        print ";", loadInst

        return loadInst


def _test():
    import doctest
    doctest.testmod()
    sys.exit()
    

def main():

    if len(sys.argv) < 2:
        _test()

    ast = compiler.parseFile(sys.argv[1])
    # print ast

    compiler.walk(ast, CodeGenLLVM())


if __name__ == '__main__':
    main()
