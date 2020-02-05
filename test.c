#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "net.h"
#include "mnist.h"
#include "cifar.h"

void printarr(double *arr, size_t n) {
  printf("[");
  for (size_t i = 0; i < n; i++) {
    printf("%.2f ", arr[i]);
  }
  printf("]\n");
}

int makeguess(double *arr, size_t n) {
  int guess;
  double max = 0.0f;
  for (size_t i = 0; i < n; i++) {
    if (arr[i] > max) {
      max = arr[i];
      guess = i;
    }
  }
  return guess;
} 

void test(NeuralNetwork_T *network, NetworkTrainingSet_T *set, uint64_t label) {

  double *output = net_feed_forward(network, set->input_set[label]);
  printf("output:   ");
  printarr(output, 10);
  printf("expected: ");
  printarr(set->label_set[label], 10);
  printf("\n");

}

int main(int argc, char *argv[]) {
  
  NetworkTrainingSet_T *cifar = cifar_make_training_set("cifar/data_batch_1.bin");
  NeuralNetwork_T *network;
  size_t h[] = {800};
  FILE *model;

  printf("finished reading CIFAR10\n");
  
  model = fopen("cifar.nn", "rb");
  network = net_from_file(model);
  
  model = fopen("cifar.nn", "wb");
  cifar->count = 100;
  cifar->epoch = 10;
  net_train(network, cifar, model);

  /*
  FILE* model;
  NeuralNetwork_T *network;
  size_t h[] ={300};
  NetworkTrainingSet_T *mnist = mnist_make_train_set("mnist_images.dat", "mnist_labels.dat");
  NetworkTrainingSet_T *mnist_test = mnist_make_train_set("mnist_test_images.dat", "mnist_test_labels.dat");

  printf("mnist set loaded.  training network...\n");
  
  model = fopen("mnist.nn", "rb");
  network = net_from_file(model);
  fclose(model);
  
  model = fopen("mnist.nn", "wb");
  mnist->count = 2000;
  mnist->epoch = 10;
  net_train(network, mnist, model);
  fclose(model);

  printf("training complete\n");
  
  mnist_free(&mnist);
  mnist_free(&mnist_test);
  net_free(&network);
  */

  int label;
  while (1) {
    scanf("%d", &label);
    test(network, cifar, label);
  }

  return 0;

}


