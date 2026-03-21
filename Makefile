all:
	$(MAKE) -C kernels/matmul

clean:
	$(MAKE) -C kernels/matmul clean

gen_tests:
	$(MAKE) -C kernels/matmul gen_tests

.PHONY: all clean gen_tests
