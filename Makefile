# Makefile for llm-serve (GGUF LLaMA inference server)

CC ?= gcc

# Detect OpenMP
OPENMP_FLAGS :=
ifeq ($(shell echo 'int main(){return 0;}' | $(CC) -fopenmp -x c - -o /dev/null 2>/dev/null && echo yes),yes)
  OPENMP_FLAGS = -fopenmp -DOMP
endif

CFLAGS  = -Ofast -march=native -Wall $(OPENMP_FLAGS)
LDFLAGS = -lm

llm-serve: serve.c llmc/gguf.h llmc/gguf_dequant.h llmc/bpe_tokenizer.h llmc/sampler.h
	$(CC) $(CFLAGS) -o $@ serve.c $(LDFLAGS)

clean:
	rm -f llm-serve

.PHONY: clean
