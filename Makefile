MAX_TIME ?= 100
ROOT ?= $(shell pwd)
TESTS ?= $(shell ls tests/*.bsv | sed 's/\.bsv//g' | sed 's/tests\///g')

all: CNN

CNN:
	-$(ROOT)/bsvbuild.sh -bs mkTb $@.bsv $(MAX_TIME)

test-%:
	-cd tests && $(ROOT)/bsvbuild.sh -bs mkTb $*.bsv $(MAX_TIME)

test: $(TESTS:%=test-%)

.PHONY: all CNN