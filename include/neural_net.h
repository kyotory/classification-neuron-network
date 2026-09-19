#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stddef.h>

#include "neural_layer.h"

#define DEFAULT_LEARNING_RATE 0.1

typedef struct {
    NeuralLayer *layers;
    int layer_count;

    // 0 - количество входов
    // 1 - количество скрытых слоёв
    // 2 - количество нейронов
    // 3 - количество выходов
    int topology[4];

    double learning_rate;
} NeuralNet;

int neural_net_init(NeuralNet *net,
                    int input_size,
                    int hidden_layers,
                    int hidden_neurons,
                    int output_size,
                    double learning_rate);

int neural_net_load(NeuralNet *net,
                    const char *filename,
                    double learning_rate);

void neural_net_free(NeuralNet *net);

int neural_net_forward(NeuralNet *net,
                       const double *input,
                       int input_size,
                       double *output);

int neural_net_train(NeuralNet *net,
                     const double *input,
                     const double *target);

int neural_net_save(const NeuralNet *net,
                    const char *filename);

size_t neural_net_memory_usage(const NeuralNet *net);

#endif