#!/usr/bin/env python

import os, sys
import argparse

from CodeGenLLVM import *

def main():

    parser = argparse.ArgumentParser(description='Python to MLIR translator')
    parser.add_argument('filename')
    parser.add_argument('--ssa', action='store_true')
    args = parser.parse_args()

    r = open(args.filename, 'r')
    mod = ast.parse(r.read())
    # print(ast.dump(mod))

    codegen = CodeGenLLVM(args.ssa)
    codegen.visit(mod)

if __name__ == '__main__':
    main()
