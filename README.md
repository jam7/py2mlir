py2llvm: A Python to LLVM translator
====================================

Author: Kazushi Marukawa  
Original Author: Syoyo Fujita


Requirements
============

  - python 3.8 or above
  - LLVM 10.0
  - llvmlite 0.39.1

Status
======

  - very early stage. Almost all features doesn't work ;-)


How to use
==========

  $ ./py2llvm <input.py>

Example
=======

Original implementation
=======================

Original py2llvm is implemented at https://code.google.com/archive/p/py2llvm.
It requires following packages.

  - python 2.4 or above
  - LLVM 2.3
  - llvm-py 0.2.1
