#include "neural_layer.h"

#include <stdlib.h>

int neural_layer_init(NeuralLayer *layer,
                      int neuron_count,
                      int input_count)
{
    int i;

    if (layer == NULL ||
        neuron_count <= 0 ||
        input_count <= 0)
        return 0;

    layer->neurons = NULL;
    layer->neuron_count = 0;

    layer->neurons =
        (Neuron *)calloc(
            neuron_count,
            sizeof(Neuron));

    if (layer->neurons == NULL)
        return 0;

    layer->neuron_count = neuron_count;

    for (i = 0; i < neuron_count; i++) {

        if (!neuron_init(
                &layer->neurons[i],
                input_count)) {

            neural_layer_free(layer);
            return 0;
        }
    }

    return 1;
}

void neural_layer_free(NeuralLayer *layer)
{
    int i;

    if (layer == NULL)
        return;

    for (i = 0;
         i < layer->neuron_count;
         i++) {

        neuron_free(
            &layer->neurons[i]);
    }

    free(layer->neurons);

    layer->neurons = NULL;
    layer->neuron_count = 0;
}

int neural_layer_forward(NeuralLayer *layer,
                         const double *inputs,
                         int input_count,
                         double *outputs)
{
    if (layer == NULL ||
        inputs == NULL ||
        outputs == NULL ||
        input_count <= 0)
        return 0;

    for (int i = 0;
         i < layer->neuron_count;
         i++) {

        outputs[i] =
            neuron_activate(
                &layer->neurons[i],
                inputs,
                input_count);
    }

    return 1;
}