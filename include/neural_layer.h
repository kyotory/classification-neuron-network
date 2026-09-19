#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include "neuron.h"

typedef struct {
    Neuron *neurons;
    int neuron_count;
} NeuralLayer;

int neural_layer_init(NeuralLayer *layer,
                      int neuron_count,
                      int input_count);

void neural_layer_free(NeuralLayer *layer);

int neural_layer_forward(NeuralLayer *layer,
                         const double *inputs,
                         int input_count,
                         double *outputs);

#endif