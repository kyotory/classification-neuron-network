#ifndef NEURON_H
#define NEURON_H

typedef struct {
    double output;
    double delta;
    double *weights;
    int weights_count;
} Neuron;

double sigmoid(double x);
double dsigmoid(double y);

int neuron_init(Neuron *neuron, int inputs);
void neuron_free(Neuron *neuron);

double neuron_activate(Neuron *neuron,
                       const double *inputs,
                       int input_count);

#endif