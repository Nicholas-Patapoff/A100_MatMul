NVCC      = nvcc
ARCH      = -arch=sm_80
CFLAGS    = -O3 -std=c++17
NVCCFLAGS = $(ARCH) $(CFLAGS)

TARGET = harness
SRCS   = harness.cu solve.cu

$(TARGET): $(SRCS) solve.h
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRCS) -lm

clean:
	rm -f $(TARGET)

generate:
	python3 generate.py

.PHONY: clean generate
