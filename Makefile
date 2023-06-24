MAX_TIME ?= 10000000
ROOT ?= $(shell pwd)
TESTS ?= $(shell ls tests/*.bsv | sed 's/\.bsv//g' | sed 's/tests\///g')
# BSC_ARGS += -vs mkTb
BSC_ARGS += -bsw mkTb
BSC_VERILOG_ARGS += -v mkTb

all: CNN

bsv-%:
	-ROOT=$(ROOT) $(ROOT)/bsvbuild.sh $(BSC_ARGS) $*.bsv $(MAX_TIME)

CNN: bsv-CNN

FC: bsv-FC

verilog-%:
	-ROOT=$(ROOT) $(ROOT)/bsvbuild.sh $(BSC_VERILOG_ARGS) $*.bsv $(MAX_TIME)

test-%:
	-cd $(ROOT)/tests && ROOT=$(ROOT) $(ROOT)/bsvbuild.sh $(BSC_ARGS) $*.bsv $(MAX_TIME)
	$(MAKE) -C $(ROOT) clean

test: $(TESTS:%=test-%)

clean:
	find . -type f \( -name "*.bo" -o -name "*.ba" \) -exec rm "{}" \;

.PHONY: all CNN clean