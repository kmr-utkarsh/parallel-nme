# PNME: A Gene-Gene Parallel Network Module Extraction Method

---

This repository contains code for the following paper:

```
Bikash Jaiswal, Kumar Utkarsh, D.K. Bhattacharyya, PNME – A gene-gene parallel network module extraction method,
Journal of Genetic Engineering and Biotechnology, Volume 16, Issue 2, 2018,
Pages 447-457, ISSN 1687-157X, https://doi.org/10.1016/j.jgeb.2018.08.003.
```

[Direct Link (Open Access)](http://www.sciencedirect.com/science/article/pii/S1687157X18300775)


---

This repository contains code for computing the GTOMm matrices from the co-expression network obtained from a microarray dataset.

## Building

The program utilizes the NVIDIA CUDA Toolkit and CUBLAS library for parallel routines. Make sure to have them installed, along with the NVIDA C compiler `nvcc`. Consult the package manager of the distribution, or download the CUDA Toolkit from the [official website](https://developer.nvidia.com/cuda-zone).

**NOTE**: The program was tested on a Linux platform running CUDA version 8.x.

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

The program utilizes only a single GPU, whichever is available first. The first card is identified by `/proc/driver/nvidia/gpus/0/` in Linux-based systems. Consult [this page](http://us.download.nvidia.com/XFree86/Linux-x86/304.132/README/procinterface.html) for documentation.

The program `cudaGTOM` will generate the modules from a corresponding co-expression matrix. The coexpression-matrix must be an undirected graph, stored as an adjacency-matrix in a CSV file.

```sh
$ ./cudaGTOM metastasis-cen-0.54.csv metastasis-module-0.54-GTOM2.csv 2
```

The parameter `m` corresponds to GTOMm, usually in the range 1-4 (GTOM1, GTOM2, GTOM3, ...).

---

## Contributors

* Kumar Utkarsh <kumarutkarsh.ingen@gmail.com>
* Bikash Jaiswal <bjjaiswal@gmail.com>

## License and Copyright

Licensed under the [GNU GPLv3](LICENSE)
