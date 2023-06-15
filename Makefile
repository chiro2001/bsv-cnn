MAX_TIME ?= 100
ROOT ?= $(shell pwd)
TESTS ?= $(shell ls tests/*.bsv | sed 's/\.bsv//g' | sed 's/tests\///g')

all: CNN

CNN:
	-ROOT=$(ROOT) $(ROOT)/bsvbuild.sh -bs mkTb $@.bsv $(MAX_TIME)

test-%:
	-cd $(ROOT)/tests && ROOT=$(ROOT) $(ROOT)/bsvbuild.sh -bs mkTb $*.bsv $(MAX_TIME)
	$(MAKE) -C $(ROOT) clean

test: $(TESTS:%=test-%)

clean:
	find . -type f \( -name "*.bo" -o -name "*.ba" \) -exec rm "{}" \;

.PHONY: all CNN clean