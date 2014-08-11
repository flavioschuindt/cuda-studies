#include <cstdio>
#include <iostream>

#include "ppm.h"
#include "utils.h"

using namespace std;

__host__ void salvaPPM( char *fname, unsigned char *buffer, int width, int height ) {

	if( !buffer ) {
		cout << "Image not saved. This ViewPoint Class in NOT FULL." << endl;
		return;
	}
	FILE *f = fopen( fname, "wb" );
	if( f == NULL ) erro( "Erro escrevendo o PPM" );

	fprintf( f, "P6\n# Gravado no curso de CUDA\n%u %u\n%u\n", width, height, 255 );
	fwrite( buffer, 3, width*height, f );
	fclose(f);

}

__host__ void lerPPM( char *fname, unsigned char **buffer, int *width, int *height ) {

	char aux[256];
	FILE *f = fopen( fname, "r" );
	if( f == NULL ) 
		erro( "Erro lendo o PPM" );

	fgets( aux, 256, f );
	fgets( aux, 256, f );
	fgets( aux, 256, f );
	sscanf( aux, "%d %d", width, height );
	fgets( aux, 256, f );

	if( !(*buffer) ) {
		free( *buffer );
	}
	
	int size = 3*(*width)*(*height)*sizeof( char );
	cout << "Dimensao da imagem: (" << *width << "," << *height <<")\n";

	if( ( *buffer = (unsigned char *)malloc( size ) ) == NULL )
		erro( "Erro lendo o buffer da imagem" );

	fread( *buffer, 3, (*width)*(*height), f );
	fclose( f );

}