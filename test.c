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

void test(NeuralNetwork_T *network, MNISTTrainSet_T *set) {
  double *out = net_feed_forward(network, set->train_set[0]);
  int guess = makeguess(out, 10);
  printf("output:   "); printarr(out, 10);
  printf("expected: "); printarr(set->label_set[0], 10);
  printf("The inputted label is %d.  The network predicts this digit is %d with %.2f%% certainty.\n",
         set->set->labels[0],
         guess,
         out[guess] * 100);
}

int main(int argc, char *argv[]) {
  
  size_t hiddens[] = {32, 32};
  NeuralNetwork_T *network = net_make(28*28, 10, hiddens, sizeof(hiddens)/sizeof(size_t));
  MNISTTrainSet_T *train = mnist_make_train_set("mnist_images.dat", "mnist_labels.dat");

  printf("MNIST set read.  beginning NN training\n");

  net_train(network, train->train_set, train->label_set, 60000, 10);
  test(network, train);

  /*
  double t0[] = {0.8, 0.8, 0.8};
  double t1[] = {0.9, 0.9, 0.9};
  double t2[] = {1.0, 1.0, 1.0};
  double *train_set[] = {t0, t1, t2};

  double l0[] = {0.8, 0.8, 0.8, 0.8};
  double l1[] = {0.9, 0.9, 0.9, 0.9};
  double l2[] = {1.0, 1.0, 1.0, 1.0};
  double *label_set[] = {l0, l1, l2};

  size_t dataset_size = 3;

  net_train(network, train_set, label_set, dataset_size, 6000);
  net_feed_forward(network, t2);
  net_print(network);
  */
  
  return 0;

}

