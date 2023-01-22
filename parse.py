#!/usr/bin/env python

import sys

import mlir
import mlir.ir as ir
from mlir.dialects import func, memref, arith, scf

r = open(sys.argv[1], 'r')

with ir.Context() as ctx:
    mod = ir.Module.parse(r.read())

# There is no ast walker in MLIR, so walking through MLIR by hand.  :-)

class Visitor:
    def visit(self, node):
        if isinstance(node, ir.Module):
            self.visit_Module(node)
        elif isinstance(node, func.FuncOp):
            self.visit_Func(node)
        elif isinstance(node, func.ReturnOp):
            self.visit_Return(node)
        elif isinstance(node, ir.Block):
            self.visit_Block(node)
        elif isinstance(node, arith.ConstantOp):
            self.visit_Constant(node)
        elif isinstance(node, arith.AddIOp):
            self.visit_Addi(node)
        elif isinstance(node, arith.SubIOp):
            self.visit_Subi(node)
        elif isinstance(node, arith.IndexCastOp):
            self.visit_IndexCast(node)
        elif isinstance(node, memref.AllocaOp):
            self.visit_Alloca(node)
        elif isinstance(node, memref.StoreOp):
            self.visit_Store(node)
        elif isinstance(node, memref.LoadOp):
            self.visit_Load(node)
        elif isinstance(node, scf.ForOp):
            self.visit_For(node)
        elif isinstance(node, scf.YieldOp):
            self.visit_Yield(node)
        else:
            print("How to do that")
            print(type(node))

    def visit_Module(self, node):
        for op in node.body.operations:
            self.visit(op)

    def visit_Func(self, node):
        assert isinstance(node, func.FuncOp)
        print("function", node.name)
        print("function_type", node.attributes["function_type"])
        for region in node.regions:
            for blk in region.blocks:
                self.visit(blk)

    def visit_Return(self, node):
        print("visit return")

    def visit_Block(self, node):
        print("visit block")
        print("  arguments")
        for arg in node.arguments:
            print("   ", arg)
        print("  statements")
        for stmt in iter(node):
            self.visit(stmt)

    def visit_Constant(self, node):
        print("visit constant")
    def visit_Addi(self, node):
        print("visit addi")
    def visit_Subi(self, node):
        print("visit subi")
    def visit_IndexCast(self, node):
        print("visit index_cast")

    def visit_Alloca(self, node):
        print("visit alloca")
    def visit_Load(self, node):
        print("visit load")
    def visit_Store(self, node):
        print("visit store")

    def visit_For(self, node):
        print("visit for")
        print("  lowerBound", node.lowerBound)
        print("  upperBound", node.upperBound)
        print("  step", node.step)
        print("  initArgs", node.initArgs)
        print("  results", node.results)
        print("  induction_variable", node.induction_variable)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Yield(self, node):
        print("visit yield")

print(mod)
visitor = Visitor()
visitor.visit(mod)
