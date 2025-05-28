#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <float.h>  // DBL_MAX
//#include <BLAS\cblas.h>

#include "dataset_io.h"

typedef struct neuralnetwork_s{
    size_t num_layer; // количество слоев
    size_t *layer_size; // массив размеров слоев
    
    double **weights; // [layer][rows * cols]  указатель на массив весов
    double **biases;  // [layer][rows] массив смещений
} NeuralNetwork;

#define REGULARIZATIONS 0.01
#define LEARNING_RATE 0.001
#define EPSILON 1e-4

// инициализирует структуру NeuralNetwork случайными значениями
// num_layer количество слоев, layer_size[num_layer] - массив размеров слоев
// в случае ошибки возврашает NULLы
NeuralNetwork* NN_Init (size_t num_layer, const size_t *layer_size);

// выполняет прямое распростронение в сети net, на вход подаються данные data
// размер которых совпадает с размером входного слоя нейронов, возврашает
// распределение вероятностей классов, размер которых совпадает с размером выходных 
// нейронов, в случае ошибки возврашает NULL
double* NN_ForwardPropagations (const uint8_t *data, const NeuralNetwork *net);

// Кросс энтропическая функция потерь, принимает на вход массив данных data, 
// интервал на которов вычесляеться ошибка [start; finish), и сеть net
// возврашает число, меру того насколько плохи тикущие параметры сети net
// в случае ошибки возврашает nun 
double NN_CrossEntropyLoss (DataSet *data, size_t start, size_t finish, NeuralNetwork *net);

// Точность оценки всех данных массива data в сети net, делает вывод точности
// по каждому примеру в формате "All correct = %d, total = %d, Current accuracy: %.2lf\r"
void NN_Accuracy (DataSet *data, NeuralNetwork *net);

// Алгоритм обратного распростронения, коректирует веса и смещения всей сети, вычеслет
// градиент функции ошибки. Параметры сети net каректируются по данным data на интервале [start; finish) 
void NN_BackPropagations (DataSet *data, size_t start, size_t finish, NeuralNetwork *net);

// Опредителить класса по признакам feature, выполняет прямое распростронение
// NN_ForwardPropagations и выдает на выходе класс с максимальной вероятностью
uint8_t NN_ClassDefinitions (uint8_t *feature, NeuralNetwork *net);

// Освобождение памяти всей структуры NeuralNetwork net
void NN_Destroy (NeuralNetwork *net);

// Сохраняет все поля структуры NeuralNetwork в бинарный файл .bin
// находяшийся в пути filename относительно того откуда запускаеться итоговый исполняемый файл
int SaveNetworkToBinaryFile(const NeuralNetwork *net, const char *filename);

// Заполняет поля структуры NeuralNetwork данными из бинарного файла .bin
// находяшийся в пути filename относительно того откуда запускаеться итоговый исполняемый файл
NeuralNetwork* LoadNetworkFromBinaryFile(const char *filename);

#endif