.PHONY: t

test:
	python -m unittest discover tests

all:
	./py2mlir.py test.py

t:
	./py2mlir.py t/arith_test.py
