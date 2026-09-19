#include "neural_net.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TARGET_ERROR 0.01
#define MAX_EPOCHES 10000

static int load_sample(const char *filename,
                       double *input,
                       int input_size,
                       double *target,
                       int output_size)
{
    FILE *file;
    char line[256];
    int i;
    int input_index = 0;
    int target_index = 0;

    file = fopen(filename, "r");

    if (file == NULL) {
        return 0;
    }

    /* Первая строка содержит целевой вектор */
    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return 0;
    }

    for (i = 0;
         line[i] != '\0' && target_index < output_size;
         i++) {

        if (line[i] == '0' || line[i] == '1') {
            target[target_index++] = line[i] - '0';
        }
    }

    if (target_index != output_size) {
        fclose(file);
        return 0;
    }

    /* Остальные строки содержат входное изображение */
    while (fgets(line, sizeof(line), file) != NULL &&
           input_index < input_size) {

        for (i = 0;
             line[i] != '\0' && input_index < input_size;
             i++) {

            if (line[i] == '0' || line[i] == '1') {
                input[input_index++] = line[i] - '0';
            }
        }
    }

    fclose(file);

    if (input_index != input_size) {
        return 0;
    }

    return 1;
}

static int collect_files(const char *folder,
                         char ***files,
                         int *count)
{
    DIR *dir;
    struct dirent *entry;
    char **list;
    int size = 0;
    int capacity = 16;

    dir = opendir(folder);

    if (dir == NULL) {
        fprintf(stderr,
                "Cannot open directory: %s\n",
                folder);
        return 0;
    }

    list = (char **)malloc(sizeof(char *) * capacity);

    if (list == NULL) {
        closedir(dir);
        return 0;
    }

    while ((entry = readdir(dir)) != NULL) {
        char *path;
        size_t length;

        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        length = strlen(folder) +
                 strlen(entry->d_name) +
                 2;

        path = (char *)malloc(length);

        if (path == NULL) {
            int i;

            for (i = 0; i < size; i++) {
                free(list[i]);
            }

            free(list);
            closedir(dir);

            return 0;
        }

        snprintf(path,
                 length,
                 "%s/%s",
                 folder,
                 entry->d_name);

        if (size >= capacity) {
            char **new_list;

            capacity *= 2;

            new_list = (char **)realloc(
                list,
                sizeof(char *) * capacity
            );

            if (new_list == NULL) {
                int i;

                free(path);

                for (i = 0; i < size; i++) {
                    free(list[i]);
                }

                free(list);
                closedir(dir);

                return 0;
            }

            list = new_list;
        }

        list[size] = path;
        size++;
    }

    closedir(dir);

    *files = list;
    *count = size;

    return 1;
}

static void free_files(char **files, int count)
{
    int i;

    if (files == NULL) {
        return;
    }

    for (i = 0; i < count; i++) {
        free(files[i]);
    }

    free(files);
}

static int argmax(const double *values, int count)
{
    int i;
    int index = 0;

    for (i = 1; i < count; i++) {
        if (values[i] > values[index]) {
            index = i;
        }
    }

    return index;
}

static const char *class_name(int class_id)
{
    switch (class_id) {
        case 0:
            return "circle";

        case 1:
            return "square";

        case 2:
            return "triangle";

        default:
            return "unknown";
    }
}

static void normalize_outputs(const double *outputs,
                              int count,
                              double *probabilities)
{
    int i;
    double sum = 0.0;

    for (i = 0; i < count; i++) {
        sum += outputs[i];
    }

    if (sum <= 0.0) {
        for (i = 0; i < count; i++) {
            probabilities[i] = 1.0 / count;
        }

        return;
    }

    for (i = 0; i < count; i++) {
        probabilities[i] = outputs[i] / sum;
    }
}

static void print_confusion_matrix(int matrix[3][3])
{
    int i;

    printf("\nConfusion matrix:\n");
    printf("          Predicted\n");
    printf("          0    1    2\n");

    for (i = 0; i < 3; i++) {
        printf("Actual %d  %3d  %3d  %3d\n",
               i,
               matrix[i][0],
               matrix[i][1],
               matrix[i][2]);
    }
}

static void print_metrics(int matrix[3][3],
                          int total)
{
    int i;
    int correct = 0;

    double accuracy;
    double precision_sum = 0.0;
    double recall_sum = 0.0;
    double f1_sum = 0.0;

    for (i = 0; i < 3; i++) {
        correct += matrix[i][i];
    }

    if (total > 0) {
        accuracy = (double)correct / total;
    } else {
        accuracy = 0.0;
    }

    for (i = 0; i < 3; i++) {
        int tp = matrix[i][i];
        int fp = 0;
        int fn = 0;
        int j;

        double precision;
        double recall;
        double f1;

        for (j = 0; j < 3; j++) {
            if (j != i) {
                fp += matrix[j][i];
                fn += matrix[i][j];
            }
        }

        if (tp + fp > 0) {
            precision = (double)tp / (tp + fp);
        } else {
            precision = 0.0;
        }

        if (tp + fn > 0) {
            recall = (double)tp / (tp + fn);
        } else {
            recall = 0.0;
        }

        if (precision + recall > 0.0) {
            f1 = 2.0 * precision * recall /
                 (precision + recall);
        } else {
            f1 = 0.0;
        }

        precision_sum += precision;
        recall_sum += recall;
        f1_sum += f1;
    }

    printf("\nAccuracy:  %.6f\n",
           accuracy);

    printf("Precision: %.6f\n",
           precision_sum / 3.0);

    printf("Recall:    %.6f\n",
           recall_sum / 3.0);

    printf("F1:        %.6f\n",
           f1_sum / 3.0);
}

static void print_usage(const char *program)
{
    printf("Usage:\n\n");

    printf("Training:\n");
    printf("  %s train <folder> <inputs> "
           "<hidden_layers> <neurons> "
           "<outputs> <learning_rate>\n",
           program);

    printf("\nPrediction:\n");
    printf("  %s predict <model> <folder>\n\n",
           program);

    printf("Example:\n");
    printf("  %s train ../data/train 49 3 10 3 0.1\n",
           program);

    printf("  %s predict ../data/model ../data/test\n",
           program);
}

static int train_network(const char *folder,
                         int input_size,
                         int hidden_layers,
                         int hidden_neurons,
                         int output_size,
                         double learning_rate)
{
    NeuralNet net;

    char **files = NULL;
    int file_count = 0;

    double *input = NULL;
    double *target = NULL;
    double *output = NULL;

    int epoch;
    int i;

    clock_t start;
    clock_t end;
    double elapsed;

    if (!collect_files(folder,
                       &files,
                       &file_count)) {
        return 0;
    }

    if (file_count == 0) {
        fprintf(stderr,
                "No training files found.\n");

        free_files(files, file_count);
        return 0;
    }

    printf("Training samples: %d\n",
           file_count);

    printf("Network: %d -> %d x %d -> %d\n",
           input_size,
           hidden_layers,
           hidden_neurons,
           output_size);

    printf("Learning rate: %.6f\n\n",
           learning_rate);

    if (!neural_net_init(&net,
                         input_size,
                         hidden_layers,
                         hidden_neurons,
                         output_size,
                         learning_rate)) {

        fprintf(stderr,
                "Neural network initialization error.\n");

        free_files(files, file_count);
        return 0;
    }

    input = (double *)malloc(
        sizeof(double) * input_size
    );

    target = (double *)malloc(
        sizeof(double) * output_size
    );

    output = (double *)malloc(
        sizeof(double) * output_size
    );

    if (input == NULL ||
        target == NULL ||
        output == NULL) {

        fprintf(stderr,
                "Memory allocation error.\n");

        free(input);
        free(target);
        free(output);

        neural_net_free(&net);
        free_files(files, file_count);

        return 0;
    }

    start = clock();

    for (epoch = 1;
         epoch <= MAX_EPOCHES;
         epoch++) {

        double error = 0.0;

        for (i = 0;
             i < file_count;
             i++) {

            if (!load_sample(files[i],
                             input,
                             input_size,
                             target,
                             output_size)) {

                fprintf(stderr,
                        "Cannot read sample: %s\n",
                        files[i]);

                free(input);
                free(target);
                free(output);

                neural_net_free(&net);
                free_files(files, file_count);

                return 0;
            }

            if (!neural_net_train(&net,
                                  input,
                                  target)) {

                fprintf(stderr,
                        "Training error.\n");

                free(input);
                free(target);
                free(output);

                neural_net_free(&net);
                free_files(files, file_count);

                return 0;
            }

            if (!neural_net_forward(&net,
                                    input,
                                    input_size,
                                    output)) {

                fprintf(stderr,
                        "Forward propagation error.\n");

                free(input);
                free(target);
                free(output);

                neural_net_free(&net);
                free_files(files, file_count);

                return 0;
            }

            {
                int j;

                for (j = 0;
                     j < output_size;
                     j++) {

                    double diff =
                        target[j] - output[j];

                    error +=
                        diff * diff / 2.0;
                }
            }
        }

        error /= file_count;

        if (epoch % 1000 == 0 ||
            error <= TARGET_ERROR) {

            printf("Epoch %d, Error: %.10f\n",
                   epoch,
                   error);
        }

        if (error <= TARGET_ERROR) {
            break;
        }
    }

    end = clock();

    elapsed =
        (double)(end - start) /
        CLOCKS_PER_SEC;

    printf("\nTraining finished.\n");

    printf("Epochs: %d\n",
           epoch);

    printf("Training CPU time: %.6f seconds\n",
           elapsed);

    printf("Memory usage: %zu bytes\n",
           neural_net_memory_usage(&net));

    if (!neural_net_save(&net,
                         "../data/model")) {

        fprintf(stderr,
                "Cannot save model.\n");

        free(input);
        free(target);
        free(output);

        neural_net_free(&net);
        free_files(files, file_count);

        return 0;
    }

    printf("Model saved to ../data/model\n");

    free(input);
    free(target);
    free(output);

    neural_net_free(&net);
    free_files(files, file_count);

    return 1;
}

static int predict_network(const char *model,
                           const char *folder)
{
    NeuralNet net;

    char **files = NULL;
    int file_count = 0;

    double *input = NULL;
    double *target = NULL;
    double *output = NULL;
    double *probabilities = NULL;

    int matrix[3][3] = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };

    int i;
    int total = 0;

    clock_t start;
    clock_t end;
    double elapsed;

    if (!neural_net_load(&net,
                         model,
                         DEFAULT_LEARNING_RATE)) {

        fprintf(stderr,
                "Cannot load model.\n");

        return 0;
    }

    if (!collect_files(folder,
                       &files,
                       &file_count)) {

        neural_net_free(&net);
        return 0;
    }

    if (file_count == 0) {
        fprintf(stderr,
                "No test files found.\n");

        neural_net_free(&net);
        free_files(files, file_count);

        return 0;
    }

    input = (double *)malloc(
        sizeof(double) * net.topology[0]
    );

    target = (double *)malloc(
        sizeof(double) * net.topology[3]
    );

    output = (double *)malloc(
        sizeof(double) * net.topology[3]
    );

    probabilities = (double *)malloc(
        sizeof(double) * net.topology[3]
    );

    if (input == NULL ||
        target == NULL ||
        output == NULL ||
        probabilities == NULL) {

        fprintf(stderr,
                "Memory allocation error.\n");

        free(input);
        free(target);
        free(output);
        free(probabilities);

        neural_net_free(&net);
        free_files(files, file_count);

        return 0;
    }

    start = clock();

    for (i = 0;
         i < file_count;
         i++) {

        int predicted;
        int actual;

        if (!load_sample(files[i],
                         input,
                         net.topology[0],
                         target,
                         net.topology[3])) {

            fprintf(stderr,
                    "Cannot read sample: %s\n",
                    files[i]);

            free(input);
            free(target);
            free(output);
            free(probabilities);

            neural_net_free(&net);
            free_files(files, file_count);

            return 0;
        }

        if (!neural_net_forward(&net,
                                input,
                                net.topology[0],
                                output)) {

            fprintf(stderr,
                    "Prediction error.\n");

            free(input);
            free(target);
            free(output);
            free(probabilities);

            neural_net_free(&net);
            free_files(files, file_count);

            return 0;
        }

        predicted = argmax(output,
                           net.topology[3]);

        actual = argmax(target,
                        net.topology[3]);

        normalize_outputs(output,
                          net.topology[3],
                          probabilities);

        printf("\n%s\n",
               files[i]);

        printf("  Actual:     %s\n",
               class_name(actual));

        printf("  Predicted:  %s\n",
               class_name(predicted));

        printf("  Circle:     %.2f%%\n",
               probabilities[0] * 100.0);

        printf("  Square:     %.2f%%\n",
               probabilities[1] * 100.0);

        printf("  Triangle:   %.2f%%\n",
               probabilities[2] * 100.0);

        if (actual >= 0 &&
            actual < 3 &&
            predicted >= 0 &&
            predicted < 3) {

            matrix[actual][predicted]++;
        }

        total++;
    }

    end = clock();

    elapsed =
        (double)(end - start) /
        CLOCKS_PER_SEC;

    printf("\nPrediction finished.\n");

    printf("Test samples: %d\n",
           total);

    printf("Prediction CPU time: %.6f seconds\n",
           elapsed);

    print_confusion_matrix(matrix);
    print_metrics(matrix, total);

    free(input);
    free(target);
    free(output);
    free(probabilities);

    neural_net_free(&net);
    free_files(files, file_count);

    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        int input_size;
        int hidden_layers;
        int hidden_neurons;
        int output_size;
        double learning_rate;

        if (argc != 8) {
            print_usage(argv[0]);
            return 1;
        }

        input_size = atoi(argv[3]);
        hidden_layers = atoi(argv[4]);
        hidden_neurons = atoi(argv[5]);
        output_size = atoi(argv[6]);
        learning_rate = atof(argv[7]);

        return train_network(argv[2],
                             input_size,
                             hidden_layers,
                             hidden_neurons,
                             output_size,
                             learning_rate)
                   ? 0
                   : 1;
    }

    if (strcmp(argv[1], "predict") == 0) {
        if (argc != 4) {
            print_usage(argv[0]);
            return 1;
        }

        return predict_network(argv[2],
                               argv[3])
                   ? 0
                   : 1;
    }

    print_usage(argv[0]);

    return 1;
}