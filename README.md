py2mlir: A Python to MLIR translator
====================================

Author: Kazushi Marukawa  
Original Author: Syoyo Fujita


Requirements
============

  - python 3.8 or above
  - LLVM 15.0.7 (working on llvmorg-15.0.7 tag)
    - MLIR python binding in llvm

Status
======

  - very early stage. Almost all features doesn't work ;-)


How to use
==========

  $ ./py2mlir.py <input.py>

Example
=======

Original implementation
=======================

Original py2llvm is implemented at https://code.google.com/archive/p/py2llvm.
It requires following packages.

  - python 2.4 or above
  - LLVM 2.3
  - llvm-py 0.2.1
