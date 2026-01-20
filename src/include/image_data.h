//
// Created by Michał on 16.01.2026.
//

#ifndef IMAGEHISTOGRAM_IMAGE_DATA_H
#define IMAGEHISTOGRAM_IMAGE_DATA_H

#include <stdint.h>

#define IMG_WIDTH  10174
#define IMG_HEIGHT 10531

extern const uint32_t hist_gray[256];
extern const char* image_filename;

int load_image(uint8_t* image);

#endif //IMAGEHISTOGRAM_IMAGE_DATA_H
