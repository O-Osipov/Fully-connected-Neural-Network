/**
 * @file neuralnetwork.c
 * @brief Реализация функций для работы с полносвязной нейронной сетью
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <float.h>  // DBL_MAX
#include <BLAS\cblas.h>

#include "dataset_io.h"
#include "neuralnetwork.h"

// вспомогательная функция для вывода массива
void print_vector(const double *vec, size_t len) {
    printf("[");
    if (len <= 6) {
        for (size_t i = 0; i < len; i++) {
            printf(" %.4f", vec[i]);
        }
    } else {
        for (size_t i = 0; i < 3; i++) {
            printf(" %.4f", vec[i]);
        }
        printf(" ...");
        for (size_t i = len - 3; i < len; i++) {
            printf(" %.4f", vec[i]);
        }
    }
    printf(" ]");
}

// вспомогательная функция для выводу внутренней структуры сети (веса и смещения по слоям)
void NN_PrintWeightsAndBiases(const NeuralNetwork *net) {
    for (size_t l = 0; l < net->num_layer; l++) {
        size_t in_size = net->layer_size[l];
        size_t out_size = net->layer_size[l + 1];

        printf("\n--- Layer %zu ---\n", l);
        printf("Weights [%zux%zu]:\n", out_size, in_size);
        for (size_t row = 0; row < out_size && row < 3; row++) {
            const double *row_ptr = &net->weights[l][row * in_size];
            print_vector(row_ptr, in_size);
            printf("\n");
        }
        if (out_size > 6) {
            printf(" ...\n");
            for (size_t row = out_size - 3; row < out_size; row++) {
                const double *row_ptr = &net->weights[l][row * in_size];
                print_vector(row_ptr, in_size);
                printf("\n");
            }
        }

        printf("Biases [%zu]: ", out_size);
        print_vector(net->biases[l], out_size);
        printf("\n");
    }
}

/**
 * @brief Функция генерации случайных чисел Xavier
 * @param input_size размер входного слоя нейронов
 * @param output_size размер выходного слоя нейронов
 */
double NN_Random (size_t input_size, size_t output_size){
    //Xavier initialization
    double limit = sqrt(6.0 / (input_size + output_size));
    return ((double)rand() / RAND_MAX) * 2 * limit - limit;
}

/**
 * @brief Функция генерации случайных чисел He
 * @param input_size размер входного слоя нейронов
 */
double NN_HeRandom(size_t input_size) {
    double stddev = sqrt(2.0 / input_size);
    double u, v, s;

    do {
        u = (double)rand() / RAND_MAX * 2.0 - 1.0;
        v = (double)rand() / RAND_MAX * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    double z = u * sqrt(-2.0 * log(s) / s);
    return z * stddev;
}

/**
 * @brief Инициализация структуры NeuralNetwork, генерация случайных весов
 * @param num_layer количество слоев сети
 * @param layer_size массив размеров нейронов на каждом слое
 * @details Выделяеться память под поля структуры NeuralNetwork
 *          Весы инициализируються функцией NN_HeRandom
 *          Смещения устанавливаються в 0
 */
NeuralNetwork* NN_Init (size_t num_layer, const size_t *layer_size){
    if (num_layer==0) {
        fprintf(stderr, "Error \"NN_Init\": num_layer mustn't be zero\n");
        return NULL;
    }
    if (!layer_size) {
        fprintf(stderr, "Error \"NN_Init\": Point layer_size mustn't be NULL\n");
        return NULL;
    }

    NeuralNetwork *net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) {
        fprintf(stderr, "Error \"NN_Init\": Failed to allocate memory for NeuralNetwork struct\n");
        return NULL;
    }

    net->num_layer = num_layer;
    net->layer_size = malloc((num_layer + 1) * sizeof(size_t));
    if (!net->layer_size){
        free(net);
        net = NULL;
        fprintf(stderr, "Error \"NN_Init\": Failed to allocate memory for layer_size\n");
        return NULL;
    }
    memcpy(net->layer_size, layer_size, (num_layer + 1) * sizeof(size_t));

    double **weights = (double**)malloc(num_layer * sizeof(double*));
    double **biases = (double**)malloc(num_layer * sizeof(double*));
    if (!weights || !biases){
        fprintf(stderr, "Error \"NN_Init\": Failed to allocate memory for array weights or biases\n");
        free(weights);
        free(biases);
        free(net->layer_size);
        free(net);
        return NULL;
    }

    for (size_t l=0; l < num_layer; l++){
        size_t column = layer_size[l];
        size_t row = layer_size[l+1];
        weights[l] = (double*)malloc(row*column * sizeof(double));
        biases[l] = (double*)malloc(row * sizeof(double));
        if (!weights[l] || !biases[l]){
            fprintf(stderr, "Error \"NN_Init\": Failed to allocate memory for weights or biases\n");
            for (size_t i=0; i<=l; i++){
                free(weights[i]);
                free(biases[i]);
            }
            free(weights);
            free(biases);
            free(net->layer_size);
            free(net);
            return NULL;
        }
        for (size_t y=0; y < row; y++){
            for (size_t x=0; x < column; x++){
                //weights[l][y*column + x] = NN_Random(column, row);
                weights[l][y*column + x] = NN_HeRandom(column);
            }
            //biases[l][y] = NN_Random(column, row);
            biases[l][y] = 0;
        }
    }
    net->weights = weights;
    net->biases = biases;

    return net;
}

/**
 * @brief Функция активации нейрона ReLU
 * @param neural значение нейрона
 */
double NN_Activation_Relu (double neural){
    //return (1)/(1+exp(neural*(-1)));
    return (neural > 0) ? neural : 0;
}

/**
 * @brief Функция активации слоя нейронов softmax
 * @param output масив значений нейронов слоя
 * @param size размер массива (слоя)
 */
int NN_Activation_Softmax(double *output, size_t size) {
    if (!output || size == 0) return 1;

    // 1. Найдём максимум, чтобы вычесть и стабилизировать экспоненты
    double max_val = -DBL_MAX;
    for (size_t i = 0; i < size; ++i) {
        if (isnan(output[i]) || isinf(output[i])) return 1;
        if (output[i] > max_val) max_val = output[i];
    }

    // 2. Вычислим сумму экспонент с вычитанием max_val
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        output[i] = exp(output[i] - max_val);
        if (isnan(output[i]) || isinf(output[i])) return 1;
        sum += output[i];
    }

    // 3. Нормализация
    if (sum == 0.0 || isnan(sum) || isinf(sum)) return 1;
    for (size_t i = 0; i < size; ++i) {
        output[i] /= sum;
    }

    return 0; // успех
}

/**
 * @brief Алгоритм прямого распростронения нейронной сети
 * @param data вектор входное слоя
 * @param net структура нейронной сети
 * @details Выполняеться матричное умножение входного слоя на веса
 *          И прибавляеться вектор смещения
 *          в многойслойной сети резулят матричных операций переходит на внутренние слои
 *          для ускорения матричных вычеслений используеться функции cblas
 */
double** NN_ForwardPropagationsWithActivations(const uint8_t *data, const NeuralNetwork *net){
    if (!data || !net) return NULL;

    double **activations = (double**)calloc((net->num_layer+1), sizeof(double*));
    if (!activations) return NULL;
    activations[0] = (double*)malloc(net->layer_size[0] * sizeof(double));

    // копирую data в массив первого слоя нейронов
    const size_t input_size = net->layer_size[0];
    double *input = (double*)malloc(input_size *sizeof(double));
    if (!input) {
        free(activations);
        return NULL;
    }    


    for (size_t i=0; i < input_size; i++){
        input[i] = (double)data[i];
        activations[0][i] = (double)data[i] / 255.0;

    }

    double *output = NULL;

    for (size_t l=0; l < net->num_layer; l++){

        size_t column = net -> layer_size[l];
        size_t row = net -> layer_size[l+1];
    
        output = (double*)malloc(row * sizeof(double));
        if (!output){
            free(input);
            return NULL;
        }

        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    row, column,
                    1.0,
                    net->weights[l], column,
                    input, 1,
                    0.0,
                    output, 1);

        // Сохраняем активации слоя
        activations[l+1] = (double*)malloc(row * sizeof(double));
        if (!activations[l]) {
            free(input);
            free(output);
            for (size_t i = 0; i < l; i++) {
                free(activations[i]);
            }
            free(activations);
            return NULL;
        }
        
        
        for (size_t i=0; i < row; i++){
            output[i] += net->biases[l][i];
            
            if (l != net->num_layer-1){
                output[i] = NN_Activation_Relu(output[i]);
            }
        }
        
        if (l == net->num_layer-1){
            if (NN_Activation_Softmax(output, row)){
                free(input);
                free(output);
                for (size_t i=0; i<l; i++){
                    free(activations[i]);
                }
                free(activations);
                return NULL;
            }
        }
        memcpy(activations[l+1], output, row * sizeof(double));

        free(input);
        input = output;
        output = NULL;
    }
    free(input);
    return activations;
}

/**
 * @brief Алгоритм прямого распростронения нейронной сети
 * @param data вектор входное слоя
 * @param net структура нейронной сети
 * @details Выполняеться матричное умножение входного слоя на веса
 *          И прибавляеться вектор смещения
 *          в многойслойной сети резулят матричных операций переходит на внутренние слои
 *          для ускорения матричных вычеслений используеться функции cblas
 */
double* NN_ForwardPropagations (const uint8_t *data, const NeuralNetwork *net){
    if (!data || !net) return NULL;

    // копирую data в массив первого слоя нейронов
    const size_t input_size = net->layer_size[0];
    double *input = (double*)malloc(input_size *sizeof(double));
    if (!input) return NULL;
    for (size_t i=0; i < input_size; i++) 
        input[i] = (double)data[i] / 255.0;

    double *output = NULL;

    for (size_t l=0; l < net->num_layer; l++){
        size_t column = net -> layer_size[l];
        size_t row = net -> layer_size[l+1];
    
        output = (double*)malloc(row * sizeof(double));
        if (!output){
            free(input);
            return NULL;
        }
        
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    row, column,
                    1.0,
                    net->weights[l], column,
                    input, 1,
                    0.0,
                    output, 1);
        
        for (size_t i=0; i < row; i++){
            output[i] += net->biases[l][i];
            if (l != net->num_layer-1){
                output[i] = NN_Activation_Relu(output[i]);
            }
        }
        
        if (l == net->num_layer-1){
            if (NN_Activation_Softmax(output, row)){
                free(input);
                free(output);
                return NULL;
            }
        }
        
        free(input);
        input = output;
        output = NULL;
    }
    return input;
}

/**
 * @brief Кросс энтропическая функция ошибки
 * @param data массив признаков примера из всего датасета
 * @param start индекс стартового примера из датасета
 * @param finish индекс конечного примера из датасета
 * @param net структура сети
 * @details на каждом примере из датасета выполняеться прямое распростронение
 *          и считаеться отрицательный логарифм вероятности правельного класс
 *          и считаеться средняя ошибка в датасете по интервалу от start до finish
 */
double NN_CrossEntropyLoss (DataSet *data, size_t start, size_t finish, NeuralNetwork *net){
    double loss = 0.0;
    size_t num_example = finish - start + 1;
    for (size_t i = start; i < finish; i++){
        const uint8_t *feature = &data->set[i][1];
        uint8_t label = data->set[i][0];
        double *predictions = NN_ForwardPropagations(feature, net);
        if (!predictions){
            printf("Error: Exit ForwardProp is NULL\n");
            return NAN;
        }
        
        // 1e-12 мальенкое смещение если вдруг predictions[label] будет равен 0
        double curr_loss = -log (predictions[label] + 1e-12);
        loss += curr_loss;
        
        free(predictions);
    }

    return loss/num_example;
}



/**
 * @brief Функция класификации картинки по признакам
 * @param feature вектор признаков картинки (цвет пикселей)
 * @param net структура нейронной сети
*/
uint8_t NN_ClassDefinitions (uint8_t *feature, NeuralNetwork *net){
    if (!feature || !net) {
        fprintf(stderr, "NN_ClassDefinitions: Invalid arguments\n");
        return -1; // ошибка
    }

    double *predictions = NN_ForwardPropagations(feature, net);
    if (!predictions){
        printf("Error: Exit ForwardProp is NULL\n");
        return -1;
    }

    uint8_t max_idx = 0;
    double max_val = predictions[0];
    for (size_t i=1; i < net->layer_size[net->num_layer]; i++){
        if (max_val < predictions[i]){
            max_val = predictions[i];
            max_idx = i;
        }
    }

    free(predictions);
    return max_idx;
}

// Процент правильных ответов сети по датасету
void NN_Accuracy (DataSet *data, NeuralNetwork *net){
    size_t correct = 0;
    size_t total = 0;
    for (size_t i=0; i < data->size; i++){
        uint8_t *feature = &data->set[i][1];
        uint8_t label = data->set[i][0];
        
        uint8_t predict_class = NN_ClassDefinitions(feature, net);
        if (predict_class == label) correct++;
        total++;
        printf("All correct = %d, total = %d, Current accuracy: %.2lf\r",correct, total, (double)correct/total);   
    }
    printf("\n");
}

/**
 * @brief Выполняет обратное распространение ошибки на участке датасета
 * 
 * @param data Массив обучающих примеров (датасет)
 * @param start Индекс первого примера в минибатче
 * @param finish Индекс последнего примера (не включая) в минибатче
 * @param net Структура нейронной сети, содержащая веса, смещения и размеры слоев
 * 
 * @details Для каждого примера в указанном диапазоне выполняется прямое распространение (forward pass),
 * вычисляется градиент ошибки с использованием кросс-энтропии и обратного распространения (backpropagation).
 * Градиенты по весам и смещениям усредняются по батчу и применяются с коэффициентом обучения LEARNING_RATE.
 * Используется метод стохастического градиентного спуска (SGD).
 */
void NN_BackPropagations (DataSet *data, size_t start, size_t finish, NeuralNetwork *net){
    clock_t time_start = clock();

    double **all_grad_w = (double**)calloc(net->num_layer, sizeof(double*));
    double **all_grad_b = (double**)calloc(net->num_layer, sizeof(double*));
    for (size_t i = 0; i < net->num_layer; i++) {
        size_t size_in = net->layer_size[i];
        size_t size_out = net->layer_size[i + 1];
        all_grad_w[i] = (double*)calloc(size_in * size_out, sizeof(double));
        all_grad_b[i] = (double*)calloc(size_out, sizeof(double));
    }

    
    for (size_t batch = start; batch<finish; batch++){
        const uint8_t *features = &data->set[batch][1];
        uint8_t label = data->set[batch][0];
        
        // значения нейронов на всех слоях (не активированные) + выход сети
        double **Activations = NN_ForwardPropagationsWithActivations(features, net);
        double **deltas = (double**)calloc(net->num_layer, sizeof(double*));
        for (size_t i = 0; i < net->num_layer; i++) {
            size_t size_in = net->layer_size[i];
            size_t size_out = net->layer_size[i + 1];
            deltas[i] = (double*)calloc(size_out, sizeof(double));
        }

        // вектор предсказаний
        double *predictions = (double*)malloc(net->layer_size[net->num_layer] * sizeof(double));
        memcpy(predictions, Activations[net->num_layer], net->layer_size[net->num_layer]*sizeof(double));
        
        for (int l = (int)net->num_layer-1; l>=0; l--){
            size_t size_in = net->layer_size[l];
            size_t size_out = net->layer_size[l+1];
            
            double *grad_w = (double*)calloc(size_out*size_in, sizeof(double));
            double *grad_b = (double*)calloc(size_out, sizeof(double));
        
            // если это последний слой
            if (l == (int)net->num_layer-1){
                double *y = (double*)calloc(size_out, sizeof(double));
                y[label] = 1;
                // //вычитаем из вектора предсказаний (Predictions) вектор эталонных предсказаний
                cblas_daxpy(size_out, -1.0, y, 1, predictions, 1);

                // сохраняем ошибку для следующих слоев
                memcpy(deltas[l], predictions, size_out * sizeof(double));

                // сохраняем градиент смещения b
                cblas_daxpy(size_out, 1.0, predictions, 1, grad_b, 1);
                
                // умножаем на вектор предыдушего слоя, на выходе получаем матрицу
                cblas_dger(CblasRowMajor,
                    size_out, size_in, 
                    1.0, 
                    predictions, 1,
                    Activations[l], 1,
                    grad_w,size_in);
                
                free(y);
            }else{
                // Обработка скрытых слоев
                double *temp = (double*)calloc(size_out, sizeof(double));

                // Матрица весов следующего слоя: weights[l+1] имеет форму [next_out x size_out]
                // Умножаем транспонированную матрицу весов на вектор дельт следующего слоя
                // temp = (W_{l+1}^T) * delta_{l+1}
                cblas_dgemv(CblasRowMajor,
                            CblasTrans,
                            net->layer_size[l+2],  // rows (next_out)
                            net->layer_size[l+1],  // cols (size_out)
                            1.0,
                            net->weights[l+1],     // weights[l+1] is [next_out][size_out]
                            net->layer_size[l+1],
                            deltas[l+1],
                            1,
                            0.0,
                            temp,
                            1);

                // Применяем производную ReLU к temp (используем z = Activations[l+1] перед ReLU)
                // temp содержит (W_{l+1}^T) * delta_{l+1}
                for (size_t j = 0; j < size_out; j++) {
                    // нужно пересчитать линейную комбинацию z = W * a + b
                    double z = 0.0;
                    for (size_t k = 0; k < size_in; k++) {
                        z += net->weights[l][j * size_in + k] * Activations[l][k];
                    }
                    z += net->biases[l][j];
                    double derivative = (z > 0.0) ? 1.0 : 0.0;
                    deltas[l][j] = temp[j] * derivative;
                }

                // Градиент смещений
                cblas_daxpy(size_out, 1.0, deltas[l], 1, grad_b, 1);

                // Градиент весов: outer product δ[l] * a[l]^T
                cblas_dger(CblasRowMajor,
                        size_out, size_in,
                        1.0,
                        deltas[l], 1,
                        Activations[l], 1,
                        grad_w, size_in);

                free(temp);
            }


            // Добавить grad_b к all_grad_b[i]
            cblas_daxpy(size_out, 1.0, grad_b, 1, all_grad_b[l], 1);
            // Добавить grad_w к all_grad_w[i]
            cblas_daxpy(size_out * size_in, 1.0, grad_w, 1, all_grad_w[l], 1);
            free(grad_w);
            free(grad_b);

            clock_t time_finish = clock();
            double total_time = (double)(time_finish - time_start) / CLOCKS_PER_SEC;    
            printf("BackProp calcus: %.2f sec\r", total_time);
        }        

        free(predictions);
        for (size_t l=0; l<net->num_layer+1; l++){
            free(Activations[l]);
        }
        free(Activations);

        for (size_t l = 0; l < net->num_layer; l++) {
            free(deltas[l]);
        }
        free(deltas);
    }

    size_t num_data = finish-start;
    double multiplier = - ((double)LEARNING_RATE / num_data);

    for (int l= (int)net->num_layer-1; l>=0; l--){
        size_t size_in = net->layer_size[l];
        size_t size_out = net->layer_size[l+1];

        cblas_daxpy(size_in*size_out, multiplier, all_grad_w[l], 1, net->weights[l], 1);
        cblas_daxpy(size_out, multiplier, all_grad_b[l], 1, net->biases[l], 1);
    }
    
    // Очистка всех выделенных градиентов
    for (size_t l = 0; l < net->num_layer; l++) {
        free(all_grad_w[l]);
        free(all_grad_b[l]);
    }
    free(all_grad_w);
    free(all_grad_b);

    clock_t time_finish = clock();
    double total_time = (double)(time_finish - time_start) / CLOCKS_PER_SEC;
    printf("BackProp calcus: %.2f sec\r", total_time);
    printf("\n");
}

// сохранение полей структуры net в бинарный файл
int SaveNetworkToBinaryFile(const NeuralNetwork *net, const char *filename) {
    if (!net || !filename) {
        fprintf(stderr, "SaveNetworkToBinaryFile: Invalid arguments\n");
        return 1;
    }

    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "SaveNetworkToBinaryFile: fopen");
        return 1;
    }

    // сохраняем число слоев
    if (fwrite(&net->num_layer, sizeof(size_t), 1, file) != 1) {
        fprintf(stderr, "Error writing num_layer\n");
        fclose(file);
        return 1;
    }

    // сохраняем массив размеров слоев
    if (fwrite(net->layer_size, sizeof(size_t), net->num_layer+1, file) != net->num_layer+1) {
        fprintf(stderr, "Error writing layer_size\n");
        fclose(file);
        return 1;
    }

    // Для каждого слоя сохраняем размеры, веса и смещения
    for (size_t l = 0; l < net->num_layer; l++) {
        size_t column = net -> layer_size[l];
        size_t row = net -> layer_size[l+1];

        if (fwrite(net->weights[l], sizeof(double), column*row, file) != column*row) {
            fprintf(stderr, "Error writing weights for layer %zu\n", l);
            fclose(file);
            return 1;
        }

        if (fwrite(net->biases[l], sizeof(double), row, file) != row) {
            fprintf(stderr, "Error writing biases for layer %zu\n", l);
            fclose(file);
            return 1;
        }
    }

    fclose(file);
    return 0;
}
// загрузка полей структуры net из бинарного файла
NeuralNetwork* LoadNetworkFromBinaryFile(const char *filename) {
    if (!filename) {
        fprintf(stderr, "LoadNetworkFromBinaryFile: Invalid arguments\n");
        return NULL;
    }

    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "LoadNetworkFromBinaryFile: fopen");
        return NULL;
    }

    NeuralNetwork *net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) {
        fprintf(stderr, "Error : Failed to allocate memory for NeuralNetwork struct\n");
        fclose(file);
        return NULL;
    }

    
    if (fread(&net->num_layer, sizeof(size_t), 1, file) != 1) {
        fprintf(stderr, "LoadNetworkFromBinaryFile: Failed to read num_layers\n");
        fclose(file);
        free(net);
        net = NULL;
        return NULL;
    }
    
    net->layer_size = malloc((net->num_layer + 1) * sizeof(size_t));
    if (!net->layer_size){
        fprintf(stderr, "Error \"NN_Init\": Failed to allocate memory for layer_size\n");
        fclose(file);
        free(net);
        net = NULL;
        return NULL;
    }
    
    if (fread(net->layer_size, sizeof(size_t), net->num_layer + 1, file) != net->num_layer + 1) {
        fprintf(stderr, "Error: Failed to read layer_size\n");
        fclose(file);
        free(net->layer_size);
        free(net);
        return NULL;
    }
    

    double **weights = (double**)malloc(net->num_layer * sizeof(double*));
    double **biases = (double**)malloc(net->num_layer * sizeof(double*));
    if (!weights || !biases){
        fprintf(stderr, "Error \"NN_Init\": Failed to allocate memory for array weights or biases\n");
        free(weights);
        weights = NULL;
        free(biases);
        biases = NULL;
        free(net->layer_size);
        net->layer_size = NULL;
        free(net);
        net = NULL;
        fclose(file);
        return NULL;
    }


    for (size_t l = 0; l < net->num_layer; l++) {
        size_t column = net->layer_size[l];
        size_t row = net->layer_size[l+1];

        weights[l] = (double*)malloc(row*column * sizeof(double));
        biases[l] = (double*)malloc(row * sizeof(double));
        if (!weights[l] || !biases[l]){
            fprintf(stderr, "Error \"NN_Init\": Failed to allocate memory for weights or biases\n");
            for (size_t i=0; i<=l; i++){
                free(weights[i]);
                weights[i]=NULL;
                free(biases[i]);
                biases[i]=NULL;
            }
            free(weights);
            weights = NULL;
            free(biases);
            biases = NULL;
            free(net->layer_size);
            net->layer_size = NULL;
            free(net);
            net = NULL;
            fclose(file);
            return NULL;
        }

        if (fread(weights[l], sizeof(double), column * row, file) != column * row) {
            fprintf(stderr, "Error \"NN_Init\": Failed to read weights[l]\n");
            for (size_t i=0; i<=l; i++){
                free(weights[i]);
                weights[i]=NULL;
                free(biases[i]);
                biases[i]=NULL;
            }
            free(weights);
            weights = NULL;
            free(biases);
            biases = NULL;
            free(net->layer_size);
            net->layer_size = NULL;
            free(net);
            net = NULL;
            fclose(file);
            return NULL;
        }

        if (fread(biases[l], sizeof(double), row, file) != row) {
            fprintf(stderr, "Error \"NN_Init\": Failed to read biases[l]\n");
            for (size_t i=0; i<=l; i++){
                free(weights[i]);
                weights[i]=NULL;
                free(biases[i]);
                biases[i]=NULL;
            }
            free(weights);
            weights = NULL;
            free(biases);
            biases = NULL;
            free(net->layer_size);
            net->layer_size = NULL;
            free(net);
            net = NULL;
            fclose(file);
            return NULL;
        }


    }
    net->weights = weights;
    net->biases = biases;

    fclose(file);
    return net;
}




/**
 * @brief Очистка струтуры NeuralNetwork 
 * @param net структура сети net
*/
void NN_Destroy (NeuralNetwork *net){
    if (!net) return;
    for (size_t l=0; l<net->num_layer; l++){
        if (net->weights[l]) {
            free(net->weights[l]);
            net->weights[l] = NULL;
        }
        if (net->biases[l]) {
            free(net->biases[l]);
            net->biases[l] = NULL;
        }
    }
    if (net->weights){
        free(net->weights);
        net->weights = NULL;
    }
    if (net->biases){
        free(net->biases);
        net->biases = NULL;
    }
    if (net->layer_size){
        free(net->layer_size);
        net->layer_size = NULL;
    }
}
