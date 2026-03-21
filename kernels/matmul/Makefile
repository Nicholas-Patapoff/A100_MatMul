NVCC      = nvcc
ARCH      = -arch=sm_80
CFLAGS    = -O3 -std=c++17
NVCCFLAGS = $(ARCH) $(CFLAGS)

KERNEL   ?= wmma_tiled
TARGET    = harness
SRCS      = harness.cu solutions/$(KERNEL).cu

$(TARGET): $(SRCS) solve.h
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRCS) -lm

clean:
	rm -f $(TARGET)

gen_tests:
	python3 gen_tests.py

.PHONY: clean gen_tests
