MPICC   = mpicc
CFLAGS  = -Wall -Wextra -O3 -std=c99 -march=native -fopenmp
LDFLAGS = -lm
TARGET  = sir_hybrid
SRC     = sir_hybrid.c
all: $(TARGET)
$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built $(TARGET)"
run:
	OMP_NUM_THREADS=2 mpirun -np 2 ./$(TARGET)
clean:
	rm -f $(TARGET) *.o
.PHONY: all run clean
