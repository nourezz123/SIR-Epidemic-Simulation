CC      = gcc
CFLAGS  = -Wall -Wextra -O3 -std=c99 -march=native
LDFLAGS = -lm
TARGET  = sir_seq
SRC     = sir_seq.c
all: $(TARGET)
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built $(TARGET)"
run: $(TARGET)
	./$(TARGET)
clean:
	rm -f $(TARGET) *.o
.PHONY: all run clean
