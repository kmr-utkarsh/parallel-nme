all: cudaGTOM

cudaGTOM: src/cudaGTOM.cu src/utils/readDataset.cu  src/utils/readDataset.h
	nvcc src/cudaGTOM.cu src/utils/readDataset.cu -o cudaGTOM -lcublas

clean:
	rm -i cudaGTOM
