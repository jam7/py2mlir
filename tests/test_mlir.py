import unittest

class TestImport(unittest.TestCase):
    def test_mlir(self):
        try:
            import mlir
        except ImportError:
            self.fail("import mlir causes ImportError")

    def test_mlir_ir(self):
        try:
            import mlir.ir
        except ImportError:
            self.fail("import mlir.ir causes ImportError")

