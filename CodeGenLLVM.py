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

llIntType     = llvm.core.Type.int()
llFloatType   = llvm.core.Type.float()
llFVec4Type   = llvm.core.Type.vector(llFloatType, 4)

def toLLVMTy(ty):

    floatTy = llvm.core.Type.float()

    d = {
          float : llFloatType
        , int   : llIntType
        , vec   : llFVec4Type
        }

    if d.has_key(ty):
        return d[ty]

    raise Exception("Unknown type:", ty)

class CodeGenLLVM:
    """
    LLVM CodeGen class 
    """

    def __init__(self):

        self.body        = ""
        self.globalscope = ""

        self.module      = llvm.core.Module.new("module")
        self.funcs       = []
        self.func        = None # Current function
        self.builder     = None

    def visitFunction(self, node):

        retLLVMTy  =    llvm.core.Type.int() # TODO

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
        funcLLVMTy = llvm.core.Type.function( retLLVMTy, argLLTys )

        func = llvm.core.Function.new( self.module, funcLLVMTy, node.name )
 
        # Assign name for each arg
        for i, name in enumerate(node.argnames):
            func.args[i].name = name

        entry = func.append_basic_block("entry")

        builder = llvm.core.Builder.new(entry)

        self.func    = func
        self.builder = builder

        self.funcs.append(func)

        self.visit(node.code)

        print self.module   # Output

    def visitStmt(self, node):

        for node in node.nodes:

            print "; [stmt]", node  

            self.visit(node)

    def visitAssign(self, node):

        if len(node.nodes) != 1:
            raise Exception("TODO:", node)

        rTy     = typer.inferType(node.expr)
        rLLInst = self.visit(node.expr)

        print "; [Asgn]. rTy = ", rTy
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
