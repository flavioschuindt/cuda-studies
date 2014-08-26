//---------------------------------------------------
// Autor: Ricardo Farias
// Data : June 2014
// Goal : Save/Read images in ppm format
//        Read returns the buffer in (r,g,b) float
//        Write expects the buffer in (r,g,b) float
//---------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <iostream>

using namespace std;


void getPPMSize( char *fname, int *width, int *height ) {

	char aux[256];
	FILE *f = fopen( fname, "r" );
	if( f == NULL ) { 
		printf( "Error read size of PPM file %s\n", fname );
		exit( 1 );
	}
	// P6
	if( fgets( aux, 256, f ) == NULL )
		cout << "Error\n";
	// Remark
	if( fgets( aux, 256, f ) == NULL ) 
		cout << "Error\n";
	// Width Height
	if( fgets( aux, 256, f ) == NULL )
		cout << "Error\n";
	sscanf( aux, "%d %d", width, height );
	fclose( f );

}



void savePPMfromFloat( char *fname, float *buffer, int width, int height ) {

	if( !buffer ) {
		cout << "Image not saved. Buffer is NULL." << endl;
		return;
	}
	FILE *f = fopen( fname, "wb" );
	if( f == NULL ) {
		printf( "Error writing PPM file. No name was provided.\n" );
		exit( 1 );
	}
	
	fprintf( f, "P6\n# Gravado no curso de CUDA\n%u %u\n%u\n", width, height, 255 );
	
	// Allocating unsigned buufer
	unsigned char *local_buffer;
	int sizeUC  = 3*width*height*sizeof( unsigned char );

	if( ( local_buffer = (unsigned char *)malloc( sizeUC ) ) == NULL ) {
		printf( "Error allocating local unsigned buffer for file %s.\n", fname );
		exit( 1 );
	}
	// Converting the buffer from float to unsigned char
	for( int i = 0 ; i < 3*width*height ; i++ ) {
		
		local_buffer[i] = (unsigned char)((int)buffer[i]);
		
	}
	
	fwrite( local_buffer, 3, width*height, f );
	fclose(f);
	free( local_buffer );
	
}


void readPPMinFloat( char *fname, float **buffer, int *width, int *height ) {

	char aux[256];
	unsigned char *local_buffer;
	FILE *f = fopen( fname, "r" );
	if( f == NULL ) { 
		printf( "Error reading PPM file %s\n", fname );
		exit( 1 );
	}

	if( fgets( aux, 256, f ) == NULL )
		cout << "Error\n";
	if( fgets( aux, 256, f ) == NULL )
		cout << "Error\n";
	if( fgets( aux, 256, f ) == NULL )
		cout << "Error\n";
	sscanf( aux, "%d %d", width, height );
	if( fgets( aux, 256, f ) == NULL )
		cout << "Error\n";
	cout << "Image " << fname << " Dimensions: (" << *width << "," << *height <<")\n";

	if( !(*buffer) ) {
		free( *buffer );
	}
	
	int sizeUC = 3*(*width)*(*height)*sizeof( unsigned char );
	int sizeF  = 3*(*width)*(*height)*sizeof( float );

	if( ( local_buffer = (unsigned char *)malloc( sizeUC ) ) == NULL ) {
		printf( "Error allocating local unsigned buffer for file %s.\n", fname );
		exit( 1 );
	}

	//void *aligned_alloc(size_t alignment, size_t size);

	//*buffer = (float *)malloc( sizeF ); // Non aligned
	//*buffer = (float *)aligned_alloc( 16, sizeF );
	posix_memalign( (void **)buffer, 16, sizeF );
	if( *buffer == NULL ) {
		printf( "Error allocating float data buffer for file %s.\n", fname );
		exit( 1 );
	}

	if( !fread( local_buffer, 3, (*width)*(*height), f ) )
		cout << "Error reding ppm\n";
	fclose( f );

	for( int i = 0 ; i < 3*(*width)*(*height) ; i++ ) {
		
		(*buffer)[i] = (float)((int)local_buffer[i]);
		
	}

	free( local_buffer );

}