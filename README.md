cuda-studies
============

General repository to study CUDA

Filter folder contains a implementation of a generic (and with no useful utility as far as i know) filter that applies some Euclidian distance in a 4-Connected Neighbourhood. It's a filter with a lot of multiplications, square roots, etc. just to make intense computing and compare results between cpu and cuda.

Usage:

make filter

filter -{c,g,s,sm} <input_image> <output_image>

-c: apply filter using only cpu

-g: apply filter using CUDA

-s: apply filter using cpu with SSE instructions

-sm: apply filter using CUDA with Shared Memory

input_image: ppm image to be processed

output_image: ppm image to be saved
