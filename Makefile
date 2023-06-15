MAX_TIME ?= 100

all: CNN

CNN:
	-./bsvbuild.sh -bs mkTb $@.bsv $(MAX_TIME)

.PHONY: all CNN