#!/usr/bin/env python

import os, sys

from CodeGenLLVM import *

def usage():
    print("Usage: py2llvm.py <input.py>")
    sys.exit(1)

def main():

    if len(sys.argv) < 2:
        usage()

    r = open(sys.argv[1], 'r')
    mod = ast.parse(r.read())
    # print(ast.dump(mod))

    codegen = CodeGenLLVM()
    codegen.visit(mod)

if __name__ == '__main__':
    main()
