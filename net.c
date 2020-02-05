#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include "net.h"

#define NN_FILE_MAGIC 0xDA6E0042

static double sig(double x) {
  return 1.0f/(1.0f + exp(-x));
}

static double dsig(double x) {
  return sig(x) * (1.0f - sig(x));
} 

static double get_correction(double learning_rate, double at_err,
                             double at_val, double from_val) {
  return learning_rate * at_err * from_val * dsig(at_val);
}

void net_print(NeuralNetwork_T *network) {
  for (size_t i = 0; i < network->layer_count; i++) {
    NetworkLayer_T *layer = &network->layers[i];
    printf("==> ");
    for (size_t j = 0; j < layer->neuron_count; j++) {
      printf("%.2f ", layer->neurons[j].value);
    }
    printf("\n");
    if (i < network->layer_count - 1) {
      for (size_t j = 0; j < layer->neuron_count; j++) {
        Neuron_T *neuron = &layer->neurons[j];
        printf("    \t[");
        for (size_t k = 0; k < neuron->axon_count; k++) {
          printf("%.2f ", neuron->axons[k].weight); 
        }
        printf("]\n");
      }
    }
  }
}

NeuralNetwork_T *net_make(size_t inputs, size_t outputs, 
    size_t *hiddens, size_t hidden_count) {

  srand(time(NULL));

  NeuralNetwork_T *network = malloc(sizeof(NeuralNetwork_T));
  assert(network);

  network->inputs = inputs;
  network->outputs = outputs;
  network->hidden_count = hidden_count;
  network->layer_count = hidden_count + 2;

  /* network makes a copy of the inputted hidden values 
   * increment each by one to include a bias */
  network->hiddens = malloc(sizeof(size_t)*hidden_count);
  assert(network->hiddens);
  for (size_t i = 0; i < hidden_count; i++) {
    network->hiddens[i] = hiddens[i];
  }

  /* network stores an array of the number of neurons in each layer */
  network->layer_counts = malloc(sizeof(size_t)*network->layer_count);
  assert(network->layer_counts);
  network->layer_counts[0] = inputs;
  network->layer_counts[network->layer_count - 1] = outputs;
  for (size_t i = 1; i < network->layer_count - 1; i++) {
    network->layer_counts[i] = network->hiddens[i - 1];
  }

  /* init layers */
  network->layers = malloc(sizeof(NetworkLayer_T)*network->layer_count);
  assert(network->layers);
  network->input_layer = &network->layers[0];
  network->output_layer = &network->layers[network->layer_count - 1];
  for (size_t i = 0; i < network->layer_count; i++) {
    NetworkLayer_T *this_layer = &network->layers[i];
    this_layer->neuron_count = network->layer_counts[i];
    this_layer->neurons = calloc(1, sizeof(Neuron_T)*network->layer_counts[i]);
    assert(this_layer->neurons);
    for (size_t j = 0; j < this_layer->neuron_count; j++) {
      this_layer->neurons[j].parent_layer = this_layer;
    }
  }

  /* init neurons and their connections */
  size_t next_layer_count;
  for (size_t i = 0; i < network->layer_count - 1; i++) {
    for (size_t j = 0; j < network->layer_counts[i]; j++) {
      Neuron_T *this_neuron = &network->layers[i].neurons[j];
      NetworkLayer_T *next_layer = &network->layers[i + 1];
      this_neuron->axons = malloc(sizeof(Axon_T)*next_layer->neuron_count);
      assert(this_neuron->axons);
      next_layer_count = (i >= 0 && i < network->layer_count - 2) ? next_layer->neuron_count - 1 
                                                                  : next_layer->neuron_count;
      this_neuron->axon_count = next_layer_count; 
      for (size_t k = 0; k < next_layer_count; k++) {
        this_neuron->axons[k].weight = ((double)rand()/RAND_MAX - 0.50)*2.0;
        this_neuron->axons[k].from = this_neuron;
        this_neuron->axons[k].to = &next_layer->neurons[k];
      }
    }
  }

  return network;
}

NeuralNetwork_T *net_copy(NeuralNetwork_T *network) {
  NeuralNetwork_T *copy = net_make(network->inputs, network->outputs,
      network->hiddens, network->hidden_count);
  for (size_t i = 0; i < copy->layer_count - 1; i++) {
    for (size_t j = 0; j < copy->layers[j].neuron_count; j++) {
      Neuron_T *neuron = &copy->layers[i].neurons[j];
      Neuron_T *template = &network->layers[i].neurons[j];
      for (size_t k = 0; k < neuron->axon_count; k++) {
        neuron->axons[k].weight = template->axons[k].weight;
      }
    }
  }

  return copy;
}

void net_free(NeuralNetwork_T **networkp) {
  assert(networkp && *networkp);

  NeuralNetwork_T *network = *networkp;

  for (size_t i = 0; i < network->layer_count; i++) {
    NetworkLayer_T *layer = &network->layers[i];
    for (size_t j = 0; j < layer->neuron_count; j++) {
      free(layer->neurons[j].axons);
    }
    free(layer->neurons);
  }

  free(network->layers);
  free(network->hiddens);
  free(network->layer_counts);
  free(network);

  *networkp = NULL;
}

void net_backprop(NeuralNetwork_T *network, double *expected) {

  double cost = 0.0f;

  /* calculate errors for output layer */
  for (size_t i = 0; i < network->outputs; i++) {
    Neuron_T *neuron = &network->output_layer->neurons[i];
    neuron->err = expected[i] - neuron->value;
    cost += neuron->err;
  }

  /* back propogate and calculate errors */
  for (size_t i = network->layer_count - 2; i > 0; i--) {
    NetworkLayer_T *layer = &network->layers[i];
    for (size_t j = 0; j < layer->neuron_count; j++) {
      Neuron_T *neuron = &layer->neurons[j];
      double my_err = 0.0f;
      for (size_t k = 0; k < neuron->axon_count; k++) {
        Axon_T *axon = &neuron->axons[k];
        my_err += axon->weight*axon->to->err; 
      }
      neuron->err = my_err;
    }
  }

  /* now correct the weights */
  double LEARNING_RATE = 0.10;
  for (size_t i = 1; i < network->layer_count; i++) {
    NetworkLayer_T *at_layer = &network->layers[i];
    NetworkLayer_T *prev_layer = &network->layers[i - 1];
    for (size_t j = 0; j < at_layer->neuron_count; j++) {
      Neuron_T *neuron = &at_layer->neurons[j];
      for (size_t k = 0; k < prev_layer->neuron_count; k++) {
        Neuron_T *neuron_from = &prev_layer->neurons[k];
        Axon_T *axon = &neuron_from->axons[j];
        axon->weight += get_correction(LEARNING_RATE, neuron->err,
                                       neuron->value, neuron_from->value);
      }
    }
  }

}

double *net_feed_forward(NeuralNetwork_T *network, double *inputs) {

  /* load the inputs into the first layer */
  for (size_t i = 0; i < network->inputs; i++) {
    network->layers[0].neurons[i].value = inputs[i];
    /*
    printf("%c ", inputs[i] > 0.30 ? 'X' : ' ');
    if (i % 28 == 0) printf("\n");
    */
  }

  double *outputs = malloc(sizeof(double)*network->outputs);
  assert(outputs);

  /* main feed forward function */
  for (size_t i = 1; i < network->layer_count; i++) {
    NetworkLayer_T *this_layer = &network->layers[i];
    NetworkLayer_T *prev_layer = &network->layers[i - 1];
    for (size_t j = 0; j < this_layer->neuron_count; j++) {
      Neuron_T *this_neuron = &this_layer->neurons[j];
      double weighted_sum = 0.0f;
      for (size_t k = 0; k < prev_layer->neuron_count; k++) {
        Neuron_T *prev_neuron = &prev_layer->neurons[k];
        weighted_sum += prev_neuron->value*prev_neuron->axons[j].weight;
      }
      this_neuron->value = sig(weighted_sum);
    }
  }

  /* load up output array */
  for (size_t i = 0; i < network->outputs; i++) {
    outputs[i] = network->output_layer->neurons[i].value;
  }

  return outputs;

}

double net_err(NeuralNetwork_T *network) {
  double sum_err = 0.0f;
  for (size_t i = 0; i < network->outputs; i++) {
    sum_err += 0.50f * pow(network->output_layer->neurons[i].err, 2);
  }
  return sum_err;
}

void net_train(NeuralNetwork_T *network, NetworkTrainingSet_T *dataset, FILE *outfile) {

  assert(dataset && dataset->count <= dataset->datacount);

  for (size_t i = 0; i < dataset->epoch; i++) {
    double mean_err = 0.0f;
    for (size_t j = 0; j < dataset->count; j++) {
      double *train_set = dataset->input_set[j];
      double *label_set = dataset->label_set[j];
      double *output = net_feed_forward(network, train_set);
      double err;
      net_backprop(network, label_set);
      mean_err += net_err(network);
      free(output);
      if (j % 1000 == 0) {
        printf("..%zu/%zu\n", j, dataset->count);
      }
    }
    mean_err /= (double)dataset->count;
    printf("epoch %zu/%zu complete. avg error %.4f\n", i + 1, dataset->epoch, mean_err);
    if (outfile) {
      if (!freopen(NULL, "wb", outfile)) {
        continue;
      }
      net_write(network, outfile);
      fflush(outfile);
      printf("wrote network to file.\n");
    }
  }

}

void net_write(NeuralNetwork_T *network, FILE *outfile) {
 
  if (!outfile) {
    return;
  }
  
  /* write magic number to file */
  uint32_t magic = NN_FILE_MAGIC;
  fwrite(&magic, sizeof(uint32_t), 1, outfile);

  /* write topology to file */
  fwrite(&network->layer_count, sizeof(size_t), 1, outfile);
  fwrite(network->layer_counts, sizeof(size_t), network->layer_count, outfile);

  /* write each neuron value, followed by outgoing weights */
  for (size_t i = 0; i < network->layer_count; i++) {
    NetworkLayer_T *layer = &network->layers[i];
    for (size_t j = 0; j < layer->neuron_count; j++) {
      Neuron_T *neuron = &layer->neurons[j];
      fwrite(&neuron->value, sizeof(double), 1, outfile);
      for (size_t k = 0; k < neuron->axon_count; k++) {
        fwrite(&neuron->axons[k].weight, sizeof(double), 1, outfile);
      }
    }
  }
  
}

NeuralNetwork_T *net_from_file(FILE *infile) {
  
  if (!infile) {
    return NULL;
  }

  uint32_t magic;
  size_t topology_count;
  size_t *topology;
  NeuralNetwork_T *network;

  fread(&magic, sizeof(uint32_t), 1, infile);
  if (magic != NN_FILE_MAGIC) {
    return NULL;
  }

  fread(&topology_count, sizeof(size_t), 1, infile);
  topology = malloc(sizeof(size_t)*topology_count);
  if (!topology) {
    return NULL;
  }

  fread(topology, sizeof(size_t), topology_count, infile);
  
  network = net_make(topology[0], topology[topology_count - 1],
                     &topology[1], topology_count - 2);
  
  for (size_t i = 0; i < topology_count; i++) {
    NetworkLayer_T *layer = &network->layers[i];
    for (size_t j = 0; j < layer->neuron_count; j++) {
      Neuron_T *neuron = &layer->neurons[j];
      fread(&neuron->value, sizeof(double), 1, infile);
      for (size_t k = 0; k < neuron->axon_count; k++) {
        fread(&neuron->axons[k].weight, sizeof(double), 1, infile);
      }
    }
  }

  free(topology);

  return network;


}
