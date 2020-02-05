#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "net.h"
#include "mnist.h"

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

void test(NeuralNetwork_T *network, MNISTTrainSet_T *set, uint64_t label) {

  double *out;
  int guess;

  if (label < set->count) {
    return;
  }

  out = net_feed_forward(network, set->train_set[label]);
  guess = makeguess(out, 10);
  printf("output:   "); printarr(out, 10);
  printf("expected: "); printarr(set->label_set[label], 10);
  printf("The inputted label is %d.  The network predicts this digit is %d with %.2f%% certainty.\n",
         set->set->labels[label],
         guess,
         out[guess] * 100);
}

int main(int argc, char *argv[]) {
  
  FILE* model = fopen("network.nn", "rb");
  MNISTTrainSet_T *mnist = mnist_make_train_set("mnist_images.dat", "mnist_labels.dat");
  MNISTTrainSet_T *mnist_test = mnist_make_train_set("mnist_test_images.dat", "mnist_test_labels.dat");
  printf("mnist sets loaded.  training network...\n");

  NeuralNetwork_T *network = net_from_file(model);
  fclose(model);
  
  net_train(network, mnist->train_set, mnist->label_set, 5000, 100, "network.nn");

  printf("training complete\n");

  int label;
  while (1) {
    scanf("%d", &label);
    test(network, mnist_test, label);
  }



  return 0;

}


