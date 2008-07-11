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

def toLLVMTy(ty):

    d = {
          float : ( llvm.core.Type.float(), llvm.core.Constant.real )
        , int   : ( llvm.core.Type.int()  , llvm.core.Constant.int  )
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

        retLLVMTy  =    llvm.core.Type.int()
        argLLVMTys =    [ llvm.core.Type.double()
                        , llvm.core.Type.double()
                        ]

        funcLLVMTy = llvm.core.Type.function( retLLVMTy, argLLVMTys )

        func = llvm.core.Function.new( self.module, funcLLVMTy, node.name )
 
        func.args[0].name = "muda1"
        func.args[1].name = "muda2"

        entry = func.append_basic_block("entry")

        builder = llvm.core.Builder.new(entry)

        self.func    = func
        self.builder = builder

        self.funcs.append(func)

        self.visit(node.code)

        print self.module

    def visitStmt(self, node):

        for node in node.nodes:

            print "[stmt]", node  

            self.visit(node)

    def visitAssign(self, node):

        if len(node.nodes) != 1:
            raise Exeption("TODO:", node)

        rTy     = typer.inferType(node.expr)
        rLLInst = self.visit(node.expr)

        print "[Asgn]. rTy = ", rTy
        print "[Asgn]. rhs = ", rLLInst

        lhsNode = node.nodes[0]

        lTy = None
        if isinstance(lhsNode, compiler.ast.AssName):

            sym = symbolTable.find(lhsNode.name)
            if sym is None:
                # The variable appears here firstly.

                # alloc storage
                (llTy, llMethod) = toLLVMTy(rTy)
                llStorage = self.builder.alloca(llTy, lhsNode.name) 

                sym = Symbol(lhsNode.name, rTy, llstorage = llStorage)
                symbolTable.append(sym)
                print "[Sym] New symbol added: ", sym



        if rTy != lTy:
            print "ERR: TypeMismatch:"

        lSym = symbolTable.find(lhsNode.name)

        storeInst = self.builder.store(rLLInst, lSym.llstorage)

        print storeInst

        print "[Asgn]", node  
        print "[Asgn] nodes = ", node.nodes 
        print "[Asgn] expr  = ", node.expr

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
        print "[AddOp] inst = ", addInst

        return addInst
 

    #
    # Leaf
    #
    def visitName(self, node):

        sym = symbolTable.lookup(node.name) 

        tmpSym = symbolTable.genUniqueSymbol(sym.type)

        # %tmp = load %name

        loadInst = self.builder.load(sym.llstorage, tmpSym.name)

        print "[Leaf] inst = ", loadInst
        return loadInst

    def visitConst(self, node):

        ty = typer.inferType(node)
        print "[Typer] %s => %s" % (str(node), str(ty))

        (llTy, llMethod) = toLLVMTy(ty)
        bufSym = symbolTable.genUniqueSymbol(ty)
        tmpSym = symbolTable.genUniqueSymbol(ty)

        # %tmp  = alloca ty
        # store ty val, %tmp
        # %inst = load ty, %tmp

        allocInst = self.builder.alloca(llTy, bufSym.name)
        llConst   = llMethod(llTy, node.value)
        storeInst = self.builder.store(llConst, allocInst)
        loadInst  = self.builder.load(allocInst, tmpSym.name)

        print loadInst

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
