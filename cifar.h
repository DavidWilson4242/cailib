#ifndef CIFAR_H
#define CIFAR_H

#include <inttypes.h>
#include "net.h"

typedef struct CIFARPixel {
  uint8_t r, g, b;
} CIFARPixel_T;

typedef struct CIFARImage {
  uint8_t label;
  size_t pixel_count;
  CIFARPixel_T *pixels;
} CIFARImage_T;

typedef struct CIFARSet {
  size_t count;
  CIFARImage_T *images;
} CIFARSet_T;

NetworkTrainingSet_T *cifar_make_training_set(const char *);

#endif
