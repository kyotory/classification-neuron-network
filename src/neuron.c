#include "neuron.h"

#include <math.h>
#include <stdlib.h>

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double dsigmoid(double y)
{
    return y * (1.0 - y);
}

int neuron_init(Neuron *neuron, int inputs)
{
    int i;

    if (neuron == NULL || inputs <= 0)
        return 0;

    neuron->output = 0.0;
    neuron->delta = 0.0;
    neuron->weights_count = inputs;

    neuron->weights =
        (double *)malloc(sizeof(double) * inputs);

    if (neuron->weights == NULL) {
        neuron->weights_count = 0;
        return 0;
    }

    for (i = 0; i < inputs; i++)
        neuron->weights[i] =
            (rand() % 100) / 100.0;

    return 1;
}

void neuron_free(Neuron *neuron)
{
    if (neuron == NULL)
        return;

    free(neuron->weights);

    neuron->weights = NULL;
    neuron->weights_count = 0;
    neuron->output = 0.0;
    neuron->delta = 0.0;
}

double neuron_activate(Neuron *neuron,
                       const double *inputs,
                       int input_count)
{
    double sum;
    int i;

    if (neuron == NULL ||
        inputs == NULL ||
        input_count <= 0)
        return 0.0;

    sum = 0.0;

    for (i = 0;
         i < neuron->weights_count &&
         i < input_count;
         i++) {

        sum +=
            neuron->weights[i] *
            inputs[i];
    }

    neuron->output = sigmoid(sum);

    return neuron->output;
}