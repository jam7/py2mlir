.PHONY: t

all:
	./py2mlir.py test.py

t:
	./py2mlir.py t/arith_test.py
