#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cifar.h"

#define CIFAR10_IMAGES_PER_BATCH 10000
#define CIFAR10_PIX_PER_IMAGE 1024

static NetworkTrainingSet_T *to_network_format(CIFARSet_T *cifar) {

  NetworkTrainingSet_T *train_set = malloc(sizeof(NetworkTrainingSet_T));
  assert(train_set);

  train_set->datacount = CIFAR10_IMAGES_PER_BATCH;
  train_set->count     = CIFAR10_IMAGES_PER_BATCH;
  train_set->epoch     = 0;
  train_set->inputdim  = CIFAR10_PIX_PER_IMAGE * 3;
  train_set->outputdim = 10;

  train_set->input_set = malloc(CIFAR10_IMAGES_PER_BATCH * sizeof(double *));
  train_set->label_set = malloc(CIFAR10_IMAGES_PER_BATCH * sizeof(double *));
  assert(train_set->input_set && train_set->label_set);

  for (size_t i = 0; i < CIFAR10_IMAGES_PER_BATCH; i++) {
    CIFARImage_T *image = &cifar->images[i];
    train_set->input_set[i] = malloc(sizeof(double) * CIFAR10_PIX_PER_IMAGE * 3);
    train_set->label_set[i] = calloc(10, sizeof(double));
    assert(train_set->input_set[i] && train_set->label_set[i]);
    train_set->label_set[i][image->label] = 1.0;
    for (size_t j = 0; j < CIFAR10_PIX_PER_IMAGE; j++) {
      train_set->input_set[i][j*3]     = (double)image->pixels[j].r/255.0;
      train_set->input_set[i][j*3 + 1] = (double)image->pixels[j].g/255.0;
      train_set->input_set[i][j*3 + 2] = (double)image->pixels[j].b/255.0;
    }
  }

  return train_set;
}

NetworkTrainingSet_T *cifar_make_training_set(const char *batchname) {

  FILE *batch = fopen(batchname, "rb");
  if (!batch) {
    fprintf(stderr, "couldn't open %s for reading\n", batchname);
    return NULL;
  }

  CIFARSet_T *cifar = malloc(sizeof(CIFARSet_T));
  assert(cifar);

  cifar->count = CIFAR10_IMAGES_PER_BATCH;
  cifar->images = malloc(CIFAR10_IMAGES_PER_BATCH * sizeof(CIFARImage_T));
  assert(cifar->images);

  for (size_t i = 0; i < CIFAR10_IMAGES_PER_BATCH; i++) {
    CIFARImage_T *image = &cifar->images[i];
    image->pixels = malloc(CIFAR10_PIX_PER_IMAGE * sizeof(CIFARPixel_T));
    assert(image->pixels);
    image->pixel_count = CIFAR10_PIX_PER_IMAGE;
    fread(&image->label, sizeof(uint8_t), 1, batch);
    for (size_t j = 0; j < CIFAR10_PIX_PER_IMAGE; j++) {
      fread(&image->pixels[j].r, sizeof(uint8_t), 1, batch);
    }
    for (size_t j = 0; j < CIFAR10_PIX_PER_IMAGE; j++) {
      fread(&image->pixels[j].g, sizeof(uint8_t), 1, batch);
    }
    for (size_t j = 0; j < CIFAR10_PIX_PER_IMAGE; j++) {
      fread(&image->pixels[j].b, sizeof(uint8_t), 1, batch);
    }
  }

  /* convert to network compatible structure */
  NetworkTrainingSet_T *train_set = to_network_format(cifar);

  /* cleanup cifar struct */

  fclose(batch);

  return train_set;

}
