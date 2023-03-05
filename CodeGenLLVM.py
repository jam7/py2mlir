#!/usr/bin/env python

import os, sys
import re
import ast

import mlir
import mlir.ir as ir
from mlir.dialects import func, arith, scf, memref

# from VecTypes import *
from MUDA import *
from TypeInference import *
from SymbolTable import *


symbolTable    = SymbolTable()
typer          = TypeInference(symbolTable)

ctx = ir.Context()
with ctx:
    i32 = ir.IntegerType.get_signless(32)
    f32 = ir.F32Type.get()
    '''
    llFVec4Type    = ll.VectorType(f32, 4)
    llFVec4PtrType = ll.PointerType(llFVec4Type)
    llIVec4Type    = ll.VectorType(i32, 4)
    '''

def toLLVMTy(ty):

    if ty is None:
        return None

    d = {
          float : f32
        , int   : i32
        , void  : None
        # , list  : list
        # , vec   : llFVec4Type
        # , void  : llVoidType
        # str   : TODO
        }

    if ty in d:
        return d[ty]

    raise Exception("Unknown type:", ty)

class CodeGenLLVM(ast.NodeVisitor):
    """
    LLVM CodeGen class
    """

    def __init__(self):

        self.body             = ""
        self.globalscope      = ""

        self.ctx = ctx

        with self.ctx, ir.Location.unknown():
          module = ir.Module.create()

        self.module           = module
        self.funcs            = []
        self.func_op          = None # Current function
        self.bb               = None

        self.currFuncRetType  = None
        self.prevFuncRetNode  = None    # for reporiting err

        self.externals        = {}

    def __del__(self):
        # TODO: find better implementation
        # Clear global symboltable to remove references to function operations
        symbolTable = SymbolTable()
        # Clear context too.
        ctx = ir.Context()

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        print("// generic_visit", ast.dump(node))

    def visit_Module(self, node):

        # emitExternalSymbols() should be called before self.visit(node.node)
        self.emitExternalSymbols()

        for stmt in node.body:
            if isinstance(stmt, ast.AST):
                self.visit(stmt)

        print(self.module)  # Output LLVM code to stdout.
        print(self.emitCommonHeader())

    def genReturnNone(self, node):
        assert self.currFuncRetType == void
        self.prevFuncRetNode = node
        with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
            ret_op = func.ReturnOp([])
        return ret_op

    def visit_Return(self, node):

        assert self.currFuncRetType is not None
        print("// Return input", ast.dump(node))

        # Return(None)
        if node.value is None:
            return self.genReturnNone(node)

        # Return(Const(None))
        if isinstance(node.value, ast.NameConstant) and node.value.value is None:
            return self.genReturnNone(node)

        ty   = typer.visit(node.value)
        print("// Return ty = ", ty)

        expr = self.visit(node.value)

        if self.currFuncRetType != ty:
            raise Exception("Different type for return expression: expected {}(lineno={}, {}) but got {}(lineno={}, {})".format(self.currFuncRetType, self.prevFuncRetNode.lineno, ast.dump(self.prevFuncRetNode), ty, node.lineno, ast.dump(node)))

        with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
            ret_op = func.ReturnOp([expr])
        return ret_op

    def mkFunctionSignature(self, retTy, node):

        # All arguments must have type hinting information.
        for arg in node.args.args:
            assert arg.annotation is not None

        argLLTys = []

        for arg in node.args.args:
            assert isinstance(arg.annotation, ast.Name)

            ty = typer.findTypeFromAName(arg.annotation.id)
            if ty is None:
                raise Exception("Unknown name of type:", arg.annotation.id)

            llTy = toLLVMTy(ty)
            argLLTys.append(llTy)

        retLLTys = []
        if retTy is not None:
            retLLTys.append(retTy)

        # funcLLVMTy = ll.FunctionType(retTy, argLLTys)
        # func = ll.Function(self.module, funcLLVMTy, node.name)
        with self.ctx, ir.Location.unknown(), ir.InsertionPoint(self.module.body):
            func_op = func.FuncOp(node.name, (argLLTys, retLLTys))

        '''
        # In MLIR, it looks like assigning name to value is not possible.

        # Assign name for each arg
        for i, name in enumerate(node.args.args):

            # if llTy == llFVec4Type:
            #     argname = name + "_p"
            # else:
            #     argname = name
            argname = name.arg

            func_op.args[i].name = argname
        '''

        return func_op

    def visit_FunctionDef(self, node):

        """
        Generate FunctionType using python type hinting information.
        """

        # init
        if node.returns is None:
            raise Exception("Type hinting of return type is required:", ast.dump(node))

        ty = typer.visit(node.returns)
        self.currFuncRetType = ty

        symbolTable.pushScope(node.name)
        retLLVMTy    = toLLVMTy(self.currFuncRetType)
        func_op      = self.mkFunctionSignature(retLLVMTy, node)
        entry        = func_op.add_entry_block()
        self.func_op = func_op
        self.bb = entry
        self.funcs.append(func_op)
        self.prevFuncRetNode = node

        '''
        # In future, try to use this value link to reduce the amount of
        # load/store...

        # Add value of each function argument to symblol table.  The value
        # is registered as entry block's arguments.
        #   "func.func"() ({
        #   ^bb0(%arg0: i32):
        #   }) {function_type = (i32) -> i32, sym_name = "test"} : () -> ()
        for i, (arg, value) in enumerate(zip(node.args.args, entry.arguments)):
            assert isinstance(value, ir.Value)
            assert isinstance(arg.annotation, ast.Name)

            ty = typer.findTypeFromAName(arg.annotation.id)
            llTy = toLLVMTy(ty)

            symbolTable.append(Symbol(arg.arg, ty, "variable", value=value))
        '''

        # Add function argument to symblol table.  Allocate memory and
        # store all values of arguments there.  The values of arguments
        # are implemented as entry block's arguments like below.
        #   "func.func"() ({
        #   ^bb0(%arg0: i32):
        #   }) {function_type = (i32) -> i32, sym_name = "test"} : () -> ()
        for i, (arg, value) in enumerate(zip(node.args.args, entry.arguments)):
            assert isinstance(value, ir.Value)
            assert isinstance(arg.annotation, ast.Name)

            ty = typer.findTypeFromAName(arg.annotation.id)
            bufSym = symbolTable.genUniqueSymbol(ty)
            llTy = toLLVMTy(ty)

            with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
                # Store all arguments in memref<llTy>.
                memref_ty = ir.MemRefType.get([], llTy)
                alloca_op = memref.AllocaOp(memref_ty, [], [])
                store_op = memref.StoreOp(value, alloca_op, [])

            symbolTable.append(Symbol(arg.arg, ty, "variable", llstorage=alloca_op))

        for stmt in node.body:
            if isinstance(stmt, ast.AST):
                self.visit(stmt)

        '''
        if self.currFuncRetType is None:
            # Add ret void.
            self.builder.ret_void()
            self.currFuncRetType = void
        '''

        symbolTable.popScope()

        # Register function to symbol table
        symbolTable.append(Symbol(node.name, self.currFuncRetType, "function", value=func_op))

    def visit_Assign(self, node):

        if len(node.targets) != 1 or isinstance(node.targets[0], ast.Tuple):
            raise Exception("TODO:", ast.dump(node))

        print("// [Asgn]", ast.dump(node))
        rTy     = typer.visit(node.value)
        print("// [Asgn]. rTy = ", rTy)

        print("// [Asgn]. node.value = ", node.value)
        rLLInst = self.visit(node.value)
        print("// [Asgn]. rhs = ", rLLInst)

        # We uses only values at this moment.
        # TODO: Support stack access to use local variables.  Need to learn
        #       memref.
        lhsNode = node.targets[0]
        return self.perform_Assign(lhsNode, rTy, rLLInst)

    def perform_Assign(self, lhsNode, rTy, rLLInst):
        lTy = None
        if isinstance(lhsNode, ast.Name):

            sym = symbolTable.find(lhsNode.id)
            if sym is None:
                # The variable appears here firstly.
                with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
                    # alloc storage
                    llTy = toLLVMTy(rTy)
                    memref_ty = ir.MemRefType.get([], llTy)
                    alloca_op = memref.AllocaOp(memref_ty, [], [])

                sym = Symbol(lhsNode.id, rTy, "variable", llstorage = alloca_op)
                symbolTable.append(sym)
                print("// [Sym] New symbol added: ", sym)

                lTy = rTy

            else:
                # symbol is already defined.
                lTy = sym.type

        if rTy != lTy:
            raise Exception("ERR: TypeMismatch: lTy = %s, rTy = %s: %s" % (lTy, rTy, ast.dump(node)))

        lSym = symbolTable.find(lhsNode.id)

        with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
            store_op = memref.StoreOp(rLLInst, lSym.llstorage, [])
        print("//", store_op)

        print("// [Asgn] target = ", lhsNode)
        print("// [Asgn] rhs = ", rLLInst)

        # No return

    def visit_AugAssign(self, node):
        assert isinstance(node.target, ast.Name)

        # Calculate target op value
        lTy = typer.visit(node.target)
        rTy = typer.visit(node.value)
        lLLInst = self.visit(node.target)
        rLLInst = self.visit(node.value)
        rLLInst = self.perform_BinOp(node.op, lTy, lLLInst, rTy, rLLInst)

        # Assign calculated value
        return self.perform_Assign(node.target, rTy, rLLInst)

    '''
    def visitIf(self, node):

        print("// ", node.tests)
        print("// ", node.else_)

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

        print("// ", node.expr)
        print("// ", node.ops[0])

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
        print("// getattr: expr", node.expr)
        print("// getattr: attrname", node.attrname)
        print("// getattr: ty", ty)

        rLLInst  = self.visit(node.expr)
        tmpSym   = symbolTable.genUniqueSymbol(ty)

        if len(node.attrname) == 1:
            # emit extract element
            s = node.attrname[0]

            inst = self.builder.extract_element(rLLInst, d[s], tmpSym.name)

        return inst
        '''

    def perform_SIToFP(self, rhs):
        with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
            inst = arith.SIToFPOp(f32, rhs)
        return inst

    def perform_IndexCast(self, rhs):
        # %2 = arith.index_cast %arg1 : index to i32
        with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
            inst = arith.IndexCastOp(i32, rhs)
        return inst

    def visit_BinOp(self, node):
        lTy = typer.visit(node.left)
        rTy = typer.visit(node.right)
        lLLInst = self.visit(node.left)
        rLLInst = self.visit(node.right)
        return self.perform_BinOp(node.op, lTy, lLLInst, rTy, rLLInst)

    def perform_BinOp(self, op, lTy, lLLInst, rTy, rLLInst):
        ty = typer.mergeType(lTy, rTy)

        if ty == float and rTy == int:
            rTy = ty
            rLLInst = self.perform_SIToFP(rLLInst)
        elif ty == float and lTy == int:
            lTy = ty
            lLLInst = self.perform_SIToFP(lLLInst)
        elif ty == int and rTy == index:
            rTy = ty
            rLLInst = self.perform_IndexCast(rLLInst)
        elif ty == int and lTy == index:
            lTy = ty
            lLLInst = self.perform_IndexCast(lLLInst)
        elif ty == float and rTy == index:
            rTy = float
            rLLInst = self.perform_IndexCast(rLLInst)
            rLLInst = self.perform_SIToFP(rLLInst)
        elif ty == float and lTy == index:
            lTy = ty
            lLLInst = self.perform_IndexCast(lLLInst)
            lLLInst = self.perform_SIToFP(lLLInst)

        if rTy != lTy:
            raise Exception("ERR: TypeMismatch: lTy = %s, rTy = %s for %s, line %d" % (lTy, rTy, ast.dump(node), node.lineno))

        tmpSym = symbolTable.genUniqueSymbol(lTy)

        with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
            if isinstance(op, ast.Add):
                if typer.isFloatType(lTy):
                    inst = arith.AddFOp(lLLInst, rLLInst)
                else:
                    inst = arith.AddIOp(lLLInst, rLLInst)
                print("// [AddOp] inst = ", inst)
            elif isinstance(op, ast.Sub):
                if typer.isFloatType(lTy):
                    inst = arith.SubFOp(lLLInst, rLLInst)
                else:
                    inst = arith.SubIOp(lLLInst, rLLInst)
                print("// [SubOp] inst = ", inst)
            elif isinstance(op, ast.Mult):
                if typer.isFloatType(lTy):
                    inst = arith.MulFOp(lLLInst, rLLInst)
                else:
                    inst = arith.MulIOp(lLLInst, rLLInst)
                print("// [MulOp] inst = ", inst)
            elif isinstance(op, ast.Div):
                if typer.isFloatType(lTy):
                    inst = arith.DivFOp(lLLInst, rLLInst)
                else:
                    raise Exception("TODO: div for type: ", lTy)
                print("// [DIvOp] inst = ", inst)
            elif isinstance(op, ast.FloorDiv):
                raise Exception("TODO: floordiv for type: ", lTy)
            elif isinstance(op, ast.Mod):
                raise Exception("TODO: mod for type: ", lTy)
            elif isinstance(op, ast.Pow):
                raise Exception("TODO: pow for type: ", lTy)
            elif isinstance(op, ast.LShift):
                raise Exception("TODO: lshift for type: ", lTy)
            elif isinstance(op, ast.RShift):
                raise Exception("TODO: rshift for type: ", lTy)
            elif isinstance(op, ast.BitOr):
                raise Exception("TODO: bitor for type: ", lTy)
            elif isinstance(op, ast.BitXor):
                raise Exception("TODO: bitxor for type: ", lTy)
            elif isinstance(op, ast.BitAnd):
                raise Exception("TODO: bitand for type: ", lTy)
            elif isinstance(op, ast.MatMult):
                raise Exception("TODO: matmult for type: ", lTy)
            else:
                raise Exception("ERROR: unknown binop: ", op)

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

    def visit_For(self, node):

        # Support only "for iv = range(expr):" statement at this momemt.

        # Check iv.
        assert isinstance(node.target, ast.Name)

        # Check range.
        assert isinstance(node.iter, ast.Call)
        assert isinstance(node.iter.func, ast.Name)
        assert node.iter.func.id == 'range'

        if len(node.iter.args) == 3:
            lb_ty = typer.visit(node.iter.args[0])
            lb = self.visit(node.iter.args[0])
            ub_ty = typer.visit(node.iter.args[1])
            ub = self.visit(node.iter.args[1])
            step_ty = typer.visit(node.iter.args[2])
            step = self.visit(node.iter.args[2])
        elif len(node.iter.args) == 2:
            lb_ty = typer.visit(node.iter.args[0])
            lb = self.visit(node.iter.args[0])
            ub_ty = typer.visit(node.iter.args[1])
            ub = self.visit(node.iter.args[1])
            step_ty = int
            step = 1
        else:
            assert len(node.iter.args) == 1
            lb, step = 0, 1
            lb_ty, step_ty = int, int
            ub_ty = typer.visit(node.iter.args[0])
            ub = self.visit(node.iter.args[0])
        assert lb_ty is int
        assert ub_ty is int
        assert step_ty is int
        if isinstance(lb, int):
            lb = self.mkLLConstInst(int, lb)
        if isinstance(step, int):
            step = self.mkLLConstInst(int, step)

        # Cast all int to index
        with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
            lb = arith.IndexCastOp(ir.IndexType.get(), lb)
            ub = arith.IndexCastOp(ir.IndexType.get(), ub)
            step = arith.IndexCastOp(ir.IndexType.get(), step)

        with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
            for_op = scf.ForOp(lb, ub, step, iter_args = [])

        # Register iv to a symbol table.
        sym = symbolTable.find(node.target.id)
        if sym is None:
            # The variable appears here firstly.
            sym = Symbol(node.target.id, index, "variable", value=for_op.induction_variable)
            symbolTable.append(sym)
            print("// [Sym] New symbol added:", sym)
            print("// [Sym] iv type:", index)
        else:
            # symbol is already defined.
            assert lb_ty == sym.type

        oldbb = self.bb
        self.bb = for_op.body
        for stmt in node.body:
            if isinstance(stmt, ast.AST):
                self.visit(stmt)

        with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
            scf.YieldOp([])

        self.bb = oldbb

    def visit_Call(self, node):

        assert isinstance(node.func, ast.Name)
        if node.func.id == "print":
            return # ignore print function calls

        print("// callfunc", ast.dump(node))

        args = [self.visit(a) for a in node.args]

        print("// callfuncafter", args)

        print("// Call ", ast.dump(node))
        print("// Call func ", node.func.id)

        '''
        print("// callfunc", node.args)

        ty = typer.findTypeFromAName(node.node.name)
        print("// callfuncafter: ty = ", ty)

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

            print("// ", args)
            c    = self.builder.call(func, args, tmp.name)

            return c
        '''
            
        #
        # Defined in the source?
        #
        ty      = typer.visit(node.func)
        funcSig = symbolTable.lookup(node.func.id)

        if funcSig.kind != "function":
            raise Exception("Symbol isn't registered as function:", node.func.id)

        # MLIR requires func_op generated from func.FuncOp().
        assert hasattr(funcSig, 'value')
        with self.ctx, ir.Location.unknown(), ir.InsertionPoint(self.bb):
            call_op = func.CallOp(funcSig.value, args)
        return call_op

    def visit_List(self, node):

        return [self.visit(a) for a in node.elts]

    #
    # Leaf
    #
    def visit_Name(self, node):

        sym = symbolTable.lookup(node.id)

        # If a node referes a storage, load it to %tmp.
        # Otherwise, use a temporary value it self.
        if hasattr(sym, 'llstorage'):
            tmpSym = symbolTable.genUniqueSymbol(sym.type)
            # %tmp = load %name
            with self.ctx, ir.InsertionPoint(self.bb), ir.Location.unknown():
                # Store all arguments in memref<llTy>.
                load_op = memref.LoadOp(sym.llstorage, [])
        else:
            load_op = sym.value

        print("// [Leaf] inst = ", load_op)
        return load_op
        '''
        # Use MLIR's value (e.g. %arg0)
        return sym.value
        '''

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
        # print("// [Typer] %s => %s" % (str(node), str(ty)))

        llTy   = toLLVMTy(ty)
        bufSym = symbolTable.genUniqueSymbol(ty)
        tmpSym = symbolTable.genUniqueSymbol(ty)

        # %tmp = arith.constant val : ty

        llConst   = None
        with self.ctx:
            with ir.InsertionPoint(self.bb), ir.Location.unknown():
                if llTy == i32:
                    val = ir.IntegerAttr.get(i32, value)
                    llConst = arith.ConstantOp(value=val, result=i32)
                elif llTy == f32:
                    val = ir.FloatAttr.get(f32, value)
                    llConst = arith.ConstantOp(value=val, result=f32)
                else:
                    print("//", value)
                    raise Exception("not supported")

        print("//", llConst)

        return llConst

    def visit_Constant(self, node):
        ty = typer.visit(node)
        return self.mkLLConstInst(ty, node.value)

    def emitCommonHeader(self):

        '''
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
        '''
        return ""

    #
    #
    #
    def emitExternalSymbols(self):

        '''
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
