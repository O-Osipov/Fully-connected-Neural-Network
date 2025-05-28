#ifndef DATASET_IO
#define DATASET_IO

#include <stdint.h>
#include <stdio.h>

#define TRAIN_SIZE 60000

// размеп картинки data set
#define PICTURE_SIZE (28*28)
// размеп картинки data set + label
#define TOTAL_PICTURE_SIZE (PICTURE_SIZE + 1)
// длина строки в dataset
#define MAX_LINE_LEN  (5 * TOTAL_PICTURE_SIZE)


typedef struct dataset_s{
    uint8_t** set;
    size_t size;
}DataSet;

// читает одну строку формату csv из потока stream
// и записывает значения в массив arr
// 0 - если без ошибок, 1 - если ошибка присутсвует 
static int read_csv(uint8_t *arr, FILE *const stream);

// читает данные из файла в file_path, размера data_size
// возврашает значения структуры DataSet
DataSet DataSet_Create (const char *file_path, size_t data_size);

// очищает поля структуры DataSet;
void DataSet_Destroy(DataSet *dataset);


#endif