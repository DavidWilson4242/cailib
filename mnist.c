#include <stdio.h>
#include <assert.h>
#include "mnist.h"

#define MNIST_LABEL_MAGIC 0x00000801
#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_SUCCESS 0
#define MNIST_ERROR 1

static void bitswap_32(uint32_t *n){ 
  uint8_t *p = (uint8_t *)n;
  uint8_t t0 = p[0], t1 = p[1];
  p[0] = p[3];
  p[1] = p[2];
  p[2] = t1;
  p[3] = t0;
}

/* returns MNIST_ERROR for error, MNIST_SUCCESS for success */
static int read_label_file(FILE *labelf, MNISTSet_T *mnist) {

  uint32_t magic;

  fread(&magic, sizeof(uint32_t), 1, labelf);
  bitswap_32(&magic);
  if (magic != MNIST_LABEL_MAGIC) {
    return MNIST_ERROR;
  }
  
  fread(&mnist->count, sizeof(uint32_t), 1, labelf);
  bitswap_32(&mnist->count);
  mnist->labels = malloc(sizeof(uint8_t)*mnist->count);
  if (!mnist->labels) {
    return MNIST_ERROR;
  }

  for (size_t i = 0; i < mnist->count; i++) {
    fread(&mnist->labels[i], sizeof(uint8_t), 1, labelf);
  }

  return MNIST_SUCCESS;

}

/* returns MNIST_ERROR for error, MNIST_SUCCESS for success */
static int read_image_file(FILE *labelf, MNISTSet_T *mnist) {
  
  uint32_t magic;
  uint32_t dimensions[3];
  uint8_t pixval;

  fread(&magic, sizeof(uint32_t), 1, labelf);
  bitswap_32(&magic);
  if (magic != MNIST_IMAGE_MAGIC) {
    return MNIST_ERROR;
  }

  fread(dimensions, sizeof(uint32_t), 3, labelf);
  for (size_t i = 0; i < 3; i++) {
    bitswap_32(&dimensions[i]);
  }
  
  mnist->images = malloc(sizeof(MNISTImage_T)*dimensions[0]);
  assert(mnist->images);

  for (size_t i = 0; i < dimensions[0]; i++) {
    MNISTImage_T *image = &mnist->images[i];
    image->cols = dimensions[1];
    image->rows = dimensions[2];
    image->pixels = malloc(sizeof(double)*image->rows*image->cols);
    assert(image->pixels);
    for (size_t j = 0; j < image->rows; j++) {
      for (size_t k = 0; k < image->cols; k++) {
        fread(&pixval, sizeof(uint8_t), 1, labelf);
        image->pixels[j*image->cols + k] = (double)pixval/255.0f;
      }
    }
  }

  return MNIST_SUCCESS;
   
}

void mnist_print_image(MNISTImage_T *image) {
  char out;
  for (size_t i = 0; i < image->rows; i++) {
    for (size_t j = 0; j < image->cols; j++) {
      out = image->pixels[i*image->cols + j] >= 0.020 ? 'X' : ' ';
      printf("%c ", out);
    }
    printf("\n");
  }
}

static MNISTSet_T *mnist_read(const char *image_file, const char *label_file) {
  
  FILE *image_handle;
  FILE *label_handle;
  MNISTSet_T *mnist = malloc(sizeof(MNISTSet_T));
  assert(mnist);

  image_handle = fopen(image_file, "rb");
  if (!image_handle) {
    return NULL;
  }

  label_handle = fopen(label_file, "rb");
  if (!label_handle) {
    return NULL;
  }
  
  if (read_label_file(label_handle, mnist) == MNIST_ERROR) {
    fclose(image_handle);
    fclose(label_handle);
    free(mnist);
    return NULL;
  }
  
  if (read_image_file(image_handle, mnist) == MNIST_ERROR) {
    fclose(image_handle);
    fclose(label_handle);
    free(mnist);
    return NULL;
  }

  fclose(image_handle);
  fclose(label_handle);

  return mnist;

}

NetworkTrainingSet_T *mnist_make_train_set(const char *image_file, const char *label_file) {
  
  MNISTSet_T *set = mnist_read("mnist_images.dat", "mnist_labels.dat");
  NetworkTrainingSet_T *train = malloc(sizeof(NetworkTrainingSet_T));
  assert(train);
  
  train->count = set->count;
  train->datacount = set->count;

  train->input_set = malloc(sizeof(double *)*set->count);
  train->label_set = malloc(sizeof(double *)*set->count);
  assert(train->input_set && train->label_set);

  for (size_t i = 0; i < set->count; i++) {
    train->input_set[i] = set->images[i].pixels;
    train->label_set[i] = calloc(sizeof(double), 10);
    assert(train->label_set[i]);
    train->label_set[i][set->labels[i]] = 1.0f;
  }

  /* mnist set struct is no longer needed... we got it into the format
   * that the network requires */
  free(set->labels);
  free(set->images);
  free(set);

  return train;

}

void mnist_free(NetworkTrainingSet_T **set) {
  
  assert(set && *set);
  
  NetworkTrainingSet_T *train = *set;
  
  /* free train&label set.  train set just references which was already
   * freed, so just free the array */
  for (size_t i = 0; i < train->datacount; i++) {
    free(train->input_set[i]);
    free(train->label_set[i]);
  }
  free(train->input_set);
  free(train->label_set);
  free(train);

  *set = NULL;

}
