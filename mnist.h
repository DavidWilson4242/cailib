#ifndef MNIST_H
#define MNIST_H

#include <stdlib.h>
#include <inttypes.h>

typedef struct MNISTImage {
  size_t cols;
  size_t rows;
  double *pixels;
} MNISTImage_T;

typedef struct MNISTSet {
  uint32_t count;
  uint8_t *labels;
  MNISTImage_T *images;
} MNISTSet_T;

typedef struct MNISTTrainSet {
  MNISTSet_T *set;
  double **train_set;
  double **label_set;
  size_t count;
} MNISTTrainSet_T;

MNISTSet_T *mnist_read(const char *, const char *);
MNISTTrainSet_T *mnist_make_train_set(const char *, const char *);
void mnist_print_image(MNISTImage_T *);

#endif
