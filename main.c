#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset_io.h"
#include "neuralnetwork.h"

// Функция для получения пути к датасету
const char* get_arg_value(int argc, char **argv, const char *flag, const char *default_value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], flag) == 0) {
            return argv[i + 1];
        }
    }
    return default_value;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage:\n");
        printf("  %s new <layer1> <layer2> ... <layerN> [--train path] [--test path]\n", argv[0]);
        printf("  %s load <model_path> [epochs] [--train path] [--test path]\n", argv[0]);
        return 1;
    }

    const char *train_path = get_arg_value(argc, argv, "--train", "./mnist/mnist_train.csv");
    const char *test_path  = get_arg_value(argc, argv, "--test",  "./mnist/mnist_test.csv");

    NeuralNetwork *net = NULL;
    size_t epochs = 1;
    size_t batch_size = 1000;

    // Загрузка датасетов
    DataSet train = DataSet_Create(train_path, 60000);
    DataSet test  = DataSet_Create(test_path, 10000);

    if (strcmp(argv[1], "new") == 0) {
        // Проверка минимального количества аргументов
        if (argc < 4) {
            printf("Usage: %s new <layer1> <layer2> ... <layerN> [--train path] [--test path]\n", argv[0]);
            return 1;
        }

        // Посчитаем количество слоёв до первого флага "--"
        int num_layers = 0;
        for (int i = 2; i < argc; ++i) {
            if (argv[i][0] == '-') break;
            num_layers++;
        }

        if (num_layers < 2) {
            printf("At least two layers required.\n");
            return 1;
        }

        size_t *layers = malloc(sizeof(size_t) * num_layers);
        if (!layers) {
            perror("malloc failed");
            return 1;
        }

        for (int i = 0; i < num_layers; ++i) {
            layers[i] = atoi(argv[i + 2]);
        }

        printf("Enter number of epochs: ");
        scanf("%zu", &epochs);

        printf("Enter mini-batch size (default = 1000): ");
        size_t input_batch;
        if (scanf("%zu", &input_batch) == 1 && input_batch > 0) {
            batch_size = input_batch;
        }

        net = NN_Init(num_layers - 1, layers);
        free(layers);

        if (!net) {
            fprintf(stderr, "Failed to initialize neural network.\n");
            return 1;
        }

    } else if (strcmp(argv[1], "load") == 0) {
        if (argc < 3) {
            printf("Usage: %s load <model_path> [epochs] [--train path] [--test path]\n", argv[0]);
            return 1;
        }

        net = LoadNetworkFromBinaryFile(argv[2]);
        if (!net) {
            fprintf(stderr, "Failed to load network from %s\n", argv[2]);
            return 1;
        }

        if (argc >= 4 && argv[3][0] != '-') {
            epochs = atoi(argv[3]);
        } else {
            printf("Enter number of epochs (default = 1): ");
            scanf("%zu", &epochs);
        }

    } else {
        printf("Unknown command: %s\n", argv[1]);
        return 1;
    }

    // Training loop
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        printf("=== Epoch %zu ===\n", epoch + 1);
        for (size_t batch = 0; batch < train.size; batch += batch_size) {
            size_t end = batch + batch_size;
            if (end > train.size) end = train.size;

            printf("Processing batch: %zu - %zu\n", batch, end);
            NN_BackPropagations(&train, batch, end, net);
            SaveNetworkToBinaryFile(net, "nn_settings.bin");
        }

        printf("Evaluation:\n");
        NN_Accuracy(&test, net);
        printf("Cross-Entropy Loss (Test)  = %lf\n", NN_CrossEntropyLoss(&test, 0, test.size, net));
        printf("Cross-Entropy Loss (Train) = %lf\n", NN_CrossEntropyLoss(&train, 0, train.size, net));
        printf("====================================\n");
    }

    // Cleanup
    NN_Destroy(net);
    DataSet_Destroy(&train);
    DataSet_Destroy(&test);

    printf("✅ Training completed.\n");
    return 0;
}
