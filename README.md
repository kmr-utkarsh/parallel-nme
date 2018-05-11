# parallel-nme: A Gene-Gene Parallel Network Module Extraction Method

---

The following is the code for the paper:

	[yet to be published]

---

This repository contains code for computing the GTOMm matrices from the co-expression network obtained from a microarray dataset.

## Building

The program utilizes the NVIDIA CUDA Toolkit and CUBLAS library for parallel routines. Make sure to have them installed, along with the NVIDA C compiler `nvcc`. Consult the package manager of the distribution, or download the CUDA Toolkit from the [official website](https://developer.nvidia.com/cuda-zone).

**NOTE**: The program was tested on a platform running CUDA version 8.x. The latest major version at this time is 9, no testing has been performed on this version.

In order to build the main executable file, run `make` in the parent directory.
Make sure `nvcc` is in the `$PATH` environment variable.

```sh
$ make
```

This will generate the binary `cudaGTOM`.

```sh
$ ./cudaGTOM
Usage: ./cudaGTOM <input file> <output file> <m>
```
The `dataset_stats` file is a shell-script used by the `cudaGTOM` program, make sure to keep them both in the same directory.

## Running

In it's current state, the program utilizes only a single GPU, whichever is available first. The first card is identified by `/proc/driver/nvidia/gpus/0/` in Linux-based systems. Consult [this page](http://us.download.nvidia.com/XFree86/Linux-x86/304.132/README/procinterface.html) for documentation.

The program `cudaGTOM` will generate the modules from a corresponding co-expression matrix. The coexpression-matrix must be an undirected graph, stored as an adjacency-matrix in a CSV file.

```sh
$ ./cudaGTOM metastasis-cen-0.54.csv metastasis-module-0.54-GTOM2.csv 2
```

The parameter `m` corresponds to GTOMm, usually in the range 1-4 (GTOM1, GTOM2, GTOM3, ...).

## Contributors

* Kumar Utkarsh <kumarutkarsh.ingen@gmail.com>
* Bikash Jaiswal <bjjaiswal@gmail.com>

## License & copyright

Licensed under the [GNU GPLv3](LICENSE)



