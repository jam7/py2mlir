====================
How to clone py2llvm
====================

Author: Kazushi Marukawa


Description
===========

Follow the mothod described at this gist, Migrate Archive Google Code SVN to
Git, at https://gist.github.com/yancyn/3f870ca6da4d4a5af5618fbd3ce4dd13.

Command lines
=============

```
$ wget https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/py2llvm/repo.svndump.gz
$ svnadmin create /tmp/repo
$ zcat repo.svndump.gz | svnadmin load /tmp/repo
<<< Started new transaction, based on original revision 1
     * editing path : branches ... done.
     * editing path : tags ... done.
     * editing path : trunk ... done.
...
$ cat > authors.txt
syoyofujita = Syoyo Fujita <syoyofujita@gmail.com>
(no author) = no_author <no_author@gmail.com>
$ git svn --stdlayout -A authors.txt clone file:///tmp/repo py2llvm
```
