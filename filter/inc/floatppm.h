//----------------------------------------------------------
// Prog: float-ppm.h
// Goal: Manipulate all ppm files using float data type as storage
// Date: August 24 2014
// By  : Ricardo Farias
//----------------------------------------------------------
#ifndef FLOATPPM_H_
#define FLOATPPM_H_

void getPPMSize( char *fname, int *width, int *height ) ;
void savePPMfromFloat( char *fname, float *buffer, int width, int height ) ;
void readPPMinFloat( char *fname, float **buffer, int *width, int *height );

#endif