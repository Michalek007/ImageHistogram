//
// Created by Michał on 16.01.2026.
//

#ifndef IMAGEHISTOGRAM_IMAGE_DATA_H
#define IMAGEHISTOGRAM_IMAGE_DATA_H

#include <stdint.h>

#define IMG_WIDTH  200
#define IMG_HEIGHT 292

extern const uint8_t image[IMG_HEIGHT][IMG_WIDTH];
extern const uint32_t hist_gray[256];

#endif //IMAGEHISTOGRAM_IMAGE_DATA_H
