import unittest
import ast
import io
from contextlib import redirect_stdout
from CodeGenLLVM import *

class TestImport(unittest.TestCase):

    def clean(self, str):
        # Remove '\n' everywhere and strip ' ' from the beggining and the end.
        return str.replace('\n', '').replace('  ', ' ').strip(' ')

    def test_empty_input(self):
        mod = ast.parse("")
        codegen = CodeGenLLVM()
        with redirect_stdout(io.StringIO()) as f:
            codegen.visit(mod)
        self.assertIn('module {\n}', f.getvalue())
        del codegen

    def test_simple_function(self):
        mod = ast.parse("def test() -> void: return")
        codegen = CodeGenLLVM()
        with redirect_stdout(io.StringIO()) as f:
            codegen.visit(mod)
        self.assertIn('''module {
  func.func @test() {
    return
  }
}''' ,f.getvalue())
        del codegen

    def test_simple_int(self):
        mod = ast.parse("def test(i: int) -> int: return i")
        codegen = CodeGenLLVM()
        with redirect_stdout(io.StringIO()) as f:
            codegen.visit(mod)
        self.assertIn('''module {
  func.func @test(%arg0: i32) -> i32 {
    %0 = memref.alloca() : memref<i32>
    memref.store %arg0, %0[] : memref<i32>
    %1 = memref.load %0[] : memref<i32>
    return %1 : i32
  }
}''' ,f.getvalue())
        del codegen

    def test_simple_float(self):
        mod = ast.parse("def test(f: float) -> float: return f")
        codegen = CodeGenLLVM()
        with redirect_stdout(io.StringIO()) as f:
            codegen.visit(mod)
        self.assertIn('''module {
  func.func @test(%arg0: f32) -> f32 {
    %0 = memref.alloca() : memref<f32>
    memref.store %arg0, %0[] : memref<f32>
    %1 = memref.load %0[] : memref<f32>
    return %1 : f32
  }
}''' ,f.getvalue())
        del codegen
