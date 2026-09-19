#include "neural_net.h"

#include <stdio.h>
#include <stdlib.h>

static void free_layers(NeuralNet *net)
{
    int i;

    if (net == NULL)
        return;

    for (i = 0; i < net->layer_count; i++)
        neural_layer_free(&net->layers[i]);

    free(net->layers);

    net->layers = NULL;
    net->layer_count = 0;
}

static int create_layers(NeuralNet *net)
{
    int input_size;
    int hidden_layers;
    int hidden_neurons;
    int output_size;
    int i;

    input_size = net->topology[0];
    hidden_layers = net->topology[1];
    hidden_neurons = net->topology[2];
    output_size = net->topology[3];

    net->layer_count = hidden_layers + 1;

    net->layers =
        (NeuralLayer *)calloc(
            net->layer_count,
            sizeof(NeuralLayer));

    if (net->layers == NULL) {
        net->layer_count = 0;
        return 0;
    }

    /*
     * Первый скрытый слой.
     */
    if (!neural_layer_init(
            &net->layers[0],
            hidden_neurons,
            input_size)) {

        free_layers(net);
        return 0;
    }

    /*
     * Остальные скрытые слои.
     */
    for (i = 1; i < hidden_layers; i++) {

        if (!neural_layer_init(
                &net->layers[i],
                hidden_neurons,
                hidden_neurons)) {

            free_layers(net);
            return 0;
        }
    }

    /*
     * Выходной слой.
     */
    if (!neural_layer_init(
            &net->layers[hidden_layers],
            output_size,
            hidden_neurons)) {

        free_layers(net);
        return 0;
    }

    return 1;
}

int neural_net_init(NeuralNet *net,
                    int input_size,
                    int hidden_layers,
                    int hidden_neurons,
                    int output_size,
                    double learning_rate)
{
    if (net == NULL)
        return 0;

    if (input_size <= 0 ||
        hidden_layers <= 0 ||
        hidden_neurons <= 0 ||
        output_size <= 0)
        return 0;

    net->layers = NULL;
    net->layer_count = 0;

    net->topology[0] = input_size;
    net->topology[1] = hidden_layers;
    net->topology[2] = hidden_neurons;
    net->topology[3] = output_size;

    net->learning_rate = learning_rate;

    return create_layers(net);
}

int neural_net_load(NeuralNet *net,
                    const char *filename,
                    double learning_rate)
{
    FILE *file;

    int i;
    int j;
    int k;

    if (net == NULL || filename == NULL)
        return 0;

    file = fopen(filename, "r");

    if (file == NULL) {
        fprintf(stderr,
                "Model file open error: %s\n",
                filename);
        return 0;
    }

    net->layers = NULL;
    net->layer_count = 0;
    net->learning_rate = learning_rate;

    for (i = 0; i < 4; i++) {

        if (fscanf(
                file,
                "%d",
                &net->topology[i]) != 1) {

            fclose(file);
            return 0;
        }
    }

    if (net->topology[0] <= 0 ||
        net->topology[1] <= 0 ||
        net->topology[2] <= 0 ||
        net->topology[3] <= 0) {

        fclose(file);
        return 0;
    }

    if (!create_layers(net)) {
        fclose(file);
        return 0;
    }

    for (i = 0;
         i < net->layer_count;
         i++) {

        NeuralLayer *layer =
            &net->layers[i];

        for (j = 0;
             j < layer->neuron_count;
             j++) {

            Neuron *neuron =
                &layer->neurons[j];

            for (k = 0;
                 k < neuron->weights_count;
                 k++) {

                if (fscanf(
                        file,
                        "%lf",
                        &neuron->weights[k]) != 1) {

                    fclose(file);
                    neural_net_free(net);
                    return 0;
                }
            }
        }
    }

    fclose(file);

    return 1;
}

void neural_net_free(NeuralNet *net)
{
    if (net == NULL)
        return;

    free_layers(net);
}

int neural_net_forward(NeuralNet *net,
                       const double *input,
                       int input_size,
                       double *output)
{
    double *current;
    double *next;

    int current_size;
    int i;

    if (net == NULL ||
        input == NULL ||
        output == NULL)
        return 0;

    if (input_size != net->topology[0])
        return 0;

    current_size = input_size;

    current =
        (double *)malloc(
            sizeof(double) * current_size);

    if (current == NULL)
        return 0;

    for (i = 0; i < current_size; i++)
        current[i] = input[i];

    for (i = 0;
         i < net->layer_count;
         i++) {

        NeuralLayer *layer =
            &net->layers[i];

        next =
            (double *)malloc(
                sizeof(double) *
                layer->neuron_count);

        if (next == NULL) {
            free(current);
            return 0;
        }

        if (!neural_layer_forward(
                layer,
                current,
                current_size,
                next)) {

            free(current);
            free(next);
            return 0;
        }

        free(current);

        current = next;
        current_size =
            layer->neuron_count;
    }

    for (i = 0;
         i < current_size;
         i++) {

        output[i] = current[i];
    }

    free(current);

    return 1;
}

int neural_net_train(NeuralNet *net,
                     const double *input,
                     const double *target)
{
    int l;
    int i;
    int j;
    int w;

    double *current;
    int current_size;

    if (net == NULL ||
        input == NULL ||
        target == NULL)
        return 0;

    /*
     * Прямое распространение.
     */
    current_size =
        net->topology[0];

    current =
        (double *)malloc(
            sizeof(double) *
            current_size);

    if (current == NULL)
        return 0;

    for (i = 0;
         i < current_size;
         i++) {

        current[i] = input[i];
    }

    for (l = 0;
         l < net->layer_count;
         l++) {

        NeuralLayer *layer =
            &net->layers[l];

        double *next =
            (double *)malloc(
                sizeof(double) *
                layer->neuron_count);

        if (next == NULL) {
            free(current);
            return 0;
        }

        if (!neural_layer_forward(
                layer,
                current,
                current_size,
                next)) {

            free(current);
            free(next);
            return 0;
        }

        free(current);

        current = next;
        current_size =
            layer->neuron_count;
    }

    free(current);

    /*
     * Delta выходного слоя.
     */
    {
        NeuralLayer *output_layer =
            &net->layers[
                net->layer_count - 1];

        for (i = 0;
             i < output_layer->neuron_count;
             i++) {

            double out =
                output_layer
                    ->neurons[i]
                    .output;

            double error =
                target[i] - out;

            output_layer
                ->neurons[i]
                .delta =
                    error *
                    dsigmoid(out);
        }
    }

    /*
     * Delta скрытых слоёв.
     */
    for (l = net->layer_count - 2;
         l >= 0;
         l--) {

        NeuralLayer *layer =
            &net->layers[l];

        NeuralLayer *next =
            &net->layers[l + 1];

        for (i = 0;
             i < layer->neuron_count;
             i++) {

            double error = 0.0;

            for (j = 0;
                 j < next->neuron_count;
                 j++) {

                error +=
                    next->neurons[j]
                        .weights[i] *
                    next->neurons[j]
                        .delta;
            }

            layer->neurons[i].delta =
                error *
                dsigmoid(
                    layer->neurons[i].output);
        }
    }

    /*
     * Обновление весов.
     */
    for (l = 0;
         l < net->layer_count;
         l++) {

        NeuralLayer *layer =
            &net->layers[l];

        for (i = 0;
             i < layer->neuron_count;
             i++) {

            Neuron *neuron =
                &layer->neurons[i];

            for (w = 0;
                 w < neuron->weights_count;
                 w++) {

                double previous_output;

                if (l == 0) {

                    previous_output =
                        input[w];

                } else {

                    previous_output =
                        net->layers[l - 1]
                            .neurons[w]
                            .output;
                }

                neuron->weights[w] +=
                    net->learning_rate *
                    neuron->delta *
                    previous_output;
            }
        }
    }

    return 1;
}

int neural_net_save(const NeuralNet *net,
                    const char *filename)
{
    FILE *file;

    int i;
    int j;
    int k;

    if (net == NULL ||
        filename == NULL)
        return 0;

    file = fopen(filename, "w");

    if (file == NULL) {
        fprintf(stderr,
                "Save file open error: %s\n",
                filename);
        return 0;
    }

    /*
     * Сохраняем топологию.
     */
    for (i = 0; i < 4; i++)
        fprintf(
            file,
            "%d ",
            net->topology[i]);

    fprintf(file, "\n");

    /*
     * Сохраняем веса.
     */
    for (i = 0;
         i < net->layer_count;
         i++) {

        const NeuralLayer *layer =
            &net->layers[i];

        for (j = 0;
             j < layer->neuron_count;
             j++) {

            const Neuron *neuron =
                &layer->neurons[j];

            for (k = 0;
                 k < neuron->weights_count;
                 k++) {

                fprintf(
                    file,
                    "%.17g ",
                    neuron->weights[k]);
            }
        }
    }

    fprintf(file, "\n");

    fclose(file);

    return 1;
}

size_t neural_net_memory_usage(
    const NeuralNet *net)
{
    size_t memory = 0;

    int i;
    int j;

    if (net == NULL)
        return 0;

    for (i = 0;
         i < net->layer_count;
         i++) {

        const NeuralLayer *layer =
            &net->layers[i];

        memory +=
            sizeof(NeuralLayer);

        memory +=
            sizeof(Neuron) *
            layer->neuron_count;

        for (j = 0;
             j < layer->neuron_count;
             j++) {

            memory +=
                sizeof(double) *
                layer->neurons[j]
                    .weights_count;
        }
    }

    return memory;
}