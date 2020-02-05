#ifndef NET_H
#define NET_H

#include <stdlib.h>

/* forward decls */
struct Neuron;
struct NetworkLayer;

typedef struct Axon {
    double weight;
    struct Neuron *from;
    struct Neuron *to;
} Axon_T;

typedef struct Neuron {
    double value;
    double err;
    size_t axon_count;
    Axon_T *axons;
    struct NetworkLayer *parent_layer;
} Neuron_T;

typedef struct NetworkLayer {
    size_t neuron_count;
    Neuron_T *neurons;
} NetworkLayer_T;

typedef struct NeuralNetwork {
    size_t inputs;
    size_t outputs;
    size_t *hiddens;
    size_t hidden_count;
    size_t layer_count;
    size_t *layer_counts;
    NetworkLayer_T *layers;
    NetworkLayer_T *input_layer;
    NetworkLayer_T *output_layer;
} NeuralNetwork_T;

NeuralNetwork_T *net_make(size_t inputs, size_t outputs, 
                          size_t *hiddens, size_t hidden_count);
NeuralNetwork_T *net_copy(NeuralNetwork_T *);
void net_free(NeuralNetwork_T **);
void net_backprop(NeuralNetwork_T *, double *);
double *net_feed_forward(NeuralNetwork_T *, double *);
double net_err(NeuralNetwork_T *);
void net_train(NeuralNetwork_T *, double **, double **, size_t, size_t, const char *);
void net_print(NeuralNetwork_T *);
void net_write(NeuralNetwork_T *, FILE *);
NeuralNetwork_T *net_from_file(FILE *);

#endif
