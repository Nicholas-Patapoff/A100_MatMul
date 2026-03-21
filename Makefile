PROBLEM ?= matmul
KERNEL  ?=

_KERNEL_ARG = $(if $(KERNEL),KERNEL=$(KERNEL),)

all:
	$(MAKE) -C kernels/$(PROBLEM) $(_KERNEL_ARG)

clean:
	$(MAKE) -C kernels/$(PROBLEM) clean

gen_tests:
	$(MAKE) -C kernels/$(PROBLEM) gen_tests

.PHONY: all clean gen_tests
