#include <cstdio>
#include <iostream>

#include "utils.h"

__host__ void erro( const char tipoDeErro[] ) {

	fprintf( stderr, "%s\n", tipoDeErro );
	exit(0);

}