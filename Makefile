#
# Makefile CPU / CUDA
#

#-----------------------------------------------------------------------------

all: 	cpu simple global shared texture htexture

cpu:	Makefile-cpu
	make -f Makefile-cpu

simple:	Makefile-simple
	make -f Makefile-simple

global:	Makefile-global
	make -f Makefile-global

shared:	Makefile-shared
	make -f Makefile-shared

texture:	Makefile-texture
	make -f Makefile-texture

htexture:	Makefile-htexture
	make -f Makefile-htexture

clean:
	#make -f Makefile-cpu      clean
	#make -f Makefile-simple   clean
	#make -f Makefile-global   clean
	make -f Makefile-shared   clean
	#make -f Makefile-texture  clean
	#make -f Makefile-htexture clean
