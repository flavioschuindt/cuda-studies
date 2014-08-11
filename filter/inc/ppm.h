//----------------------------------------------------------
// Prog: ppm.h
// Goal: Manipulate all ppm files
// Date: August 03 2014
// By  : Flávio Schuindt (Based on Prof. Ricardo Farias work)
//----------------------------------------------------------
#ifndef PPM_H_
#define PPM_H_

__host__ void salvaPPM( char *fname, unsigned char *buffer, int width, int height );
__host__ void lerPPM( char *fname, unsigned char **buffer, int *width, int *height );

#endif