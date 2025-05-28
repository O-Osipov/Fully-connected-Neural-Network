#include "dataset_io.h"

#include <string.h>
#include <stdlib.h>

// читает одну строку формату csv из потока stream
// и записывает значения в массив arr
// 0 - если без ошибок, 1 - если ошибка присутсвует 
static int read_csv(uint8_t *arr, FILE *const stream){ 
    char buffer[MAX_LINE_LEN];
    if (fgets(buffer, sizeof(buffer), stream) == NULL){
        fprintf(stderr, "Error read csv: Can't read string from stream\n");
        return 1;
    }

    int idx = 0;
    char *token = strtok(buffer, ",");
    while (token!=NULL && idx < TOTAL_PICTURE_SIZE){
        char *str_val;
        long val = strtol(token, &str_val, 10);
        if (val < 0 || val > 255){
            fprintf(stderr, "Error read csv: Value out of range: %ld\n", val);
            return 1;
        }

        arr[idx++] = (uint8_t)val;
        token = strtok(NULL, ",");
    }

    if (idx != TOTAL_PICTURE_SIZE) {
        fprintf(stderr, "Error read csv: Incorrect number of fields: %d (expected %d)\n", idx, TOTAL_PICTURE_SIZE);
        return 1;
    }

    return 0;
}   

DataSet DataSet_Create (const char *file_path, size_t data_size){
    DataSet dataset;
    dataset.set = NULL;
    dataset.size = 0;

    if (!file_path || data_size == 0) {
        fprintf(stderr, "Error DataSet_Create: Invalid arguments to DataSet_Create.\n");
        return dataset;
    }

    FILE *const file = fopen(file_path, "r");
    if (!file){
        fprintf(stderr, "Error DataSet_Create: Do not open file stream\n");
        return dataset;  
    }

    dataset.set = (uint8_t**)calloc(data_size, sizeof(uint8_t*));
    if (!dataset.set){
        fprintf(stderr, "Error DataSet_Create: Memory allocation failed for dataset.set");
        fclose(file);
        return dataset;
    }

    for (size_t i = 0; i < data_size; ++i) {
        dataset.set[i] = malloc(TOTAL_PICTURE_SIZE * sizeof(uint8_t));
        if (!dataset.set[i]) {
            fprintf(stderr, "Error DataSet_Create:  Memory allocation failed for row");

            // Очистка уже выделенного
            for (size_t j = 0; j < i; ++j)
                free(dataset.set[j]);
            free(dataset.set);
            fclose(file);
            dataset.set = NULL;
            return dataset;
        }

        if (read_csv(dataset.set[i], file)) {
            fprintf(stderr, "Error DataSet_Create: Failed to read line %lld\n", i + 1);

            for (size_t j = 0; j <= i; ++j)
                free(dataset.set[j]);
            free(dataset.set);
            fclose(file);
            dataset.set = NULL;
            return dataset;
        }

        fprintf(stdin, "read %d line\r", i);
    }
    fclose(file);
    dataset.size = data_size;

    return dataset;
}

void DataSet_Destroy(DataSet *dataset){
    if (!dataset || !dataset->set)
        return;
    for (size_t i=0; i<dataset->size; i++){
        free(dataset->set[i]);
    }
    free(dataset->set);
    dataset->set = NULL;
    dataset->size = 0;
}

