#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 100000

float * readDataset(int *rows, int *cols, char *filename);
int printDataset(float *matrix, int rows, int cols);
int writeDataset(char *filename, float *matrix, int rows, int cols);
