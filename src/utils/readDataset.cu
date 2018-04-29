#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "readDataset.h"
#define MAX_LINE_LENGTH 100000

float * readDataset(int *rows, int *cols, char *filename) {

    char elem;
    int row = 0, col = 0, index = 0;
    unsigned long int global_index = 0;
    FILE *fp = fopen(filename, "r");
    if(fp == NULL) {
        printf("ERROR: Cannot open file: %s", filename);
        return NULL;
    }

    // Determine Size of Dataset
    char command[50], stats[10];
    strcpy(command, "./dataset_stats ");
    strcat(command, filename);
    FILE *pl = popen(command, "r");
    if(pl == NULL) {
        printf("ERROR: Script 'dataset_stats' missing");
        return NULL;
    }
    fgets(stats, sizeof(stats) - 1, pl);
    *rows = (int) strtol(stats, NULL, 10);
    fgets(stats, sizeof(stats) - 1, pl);
    *cols = (int) strtol(stats, NULL, 10);
    fclose(pl);

    int width = *rows;
    int height = *cols;

    float *matrix = (float *) malloc(sizeof(float) * width * height);

    size_t size = MAX_LINE_LENGTH;
    char *lineptr = (char *) malloc(size * sizeof(char));

    while(getline(&lineptr, &size, fp) > 0) {

        col = 0;
        for(index = 0; index < strlen(lineptr); index++) {
            elem = lineptr[index];
            if(elem == '0') {
                *(matrix + row * height + col) = 0.0;
                col++;
            }
            if(elem == '1') {
                *(matrix + row * height + col) = 1.0;
                col++;
            }
            global_index++;
        }
        row++;
    }

    fclose(fp);
    return matrix;
}

int printDataset(float *matrix, int rows, int cols) {

    int row, col;
    for(row = 0; row < rows; row++) {
        for(col = 0; col < cols; col++) {
            printf("%f ", *(matrix + row * cols + col));
        }
        printf("\n");
    }
    return 0;
}


int writeDataset(char *filename, float *matrix, int rows, int cols) {

    FILE *fp = fopen(filename, "w");
    char line[1000000];
    char temp[15];
    int row, col;

    for(row = 0; row < rows; row++) {
        for(col = 0; col < cols - 1; col++) {
	    memset(temp, 0x0, sizeof(temp));
            sprintf(temp, "%lu", (unsigned long) *(matrix + row * cols + col));
	    strcat(line, temp);
	    strcat(line, ",");
        }

     	sprintf(temp, "%lu", (unsigned long) *(matrix + row * cols + col));
	strcat(line, temp);

        fprintf(fp, "%s", line);
	fprintf(fp, "\r\n");
	memset(line, 0x0, sizeof(line));
    }
    fclose(fp);
    return 0;
}

