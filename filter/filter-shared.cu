//-----------------------------------------
// Autor: Flávio Schuindt
// Data : Ago 2014
// Goal : Apply some filter in image using shared memory and thread blocks
//-----------------------------------------

/***************************************************************************************************
	Includes
***************************************************************************************************/

#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <string.h>

#include "rf-time.h"
#include "utils.h"
#include "ppm.h"

/***************************************************************************************************
	Defines
***************************************************************************************************/
#define CUDA_SAFE_CALL

#define CPU "-c"
#define GPU "-g"
#define SSE "-s"
#define TILE_WIDTH 32
#define ELEM(i,j,DIMX_) ((i)+(j)*(DIMX_))

/***************************************************************************************************
	Functions
***************************************************************************************************/

using namespace std;

/**************************************************************************************************/

__global__ void filterGPU( unsigned char *image1, 
						   unsigned char *res, 
						   int width, 
						   int height ){

	__shared__ unsigned char tile[TILE_WIDTH*TILE_WIDTH*3];

	int ct = threadIdx.x; // Coluna do tile
	int lt = threadIdx.y; // Linha do tile

	// Calcula o índice da imagem de entrada
	int row = blockIdx.y*(TILE_WIDTH)+lt;
	int col = blockIdx.x*(TILE_WIDTH)+ct;

	// Coeficientes para o filtro
	float a1, a2, a3, a4;

	// a é o elemento corrente. b, c, d, e são os seus vizinhos 4-conectividade.
	int a, b, c, d, e;

	if( row < height && col < width )
	{
		
		// Copia colaborativa para calculo de um tile
		int idx = 3*ELEM( col, row, width );
		if (blockIdx.x == 0)
		{
			tile[3*(lt*TILE_WIDTH+ct)] = image1[ idx ];
			tile[3*(lt*TILE_WIDTH+ct)+1] = image1[ idx+1 ];
			tile[3*(lt*TILE_WIDTH+ct)+2] = image1[ idx+2 ];
		}
		else
		{
			idx = idx - 2;
			tile[3*(lt*TILE_WIDTH+ct)] = image1[ idx ];
			tile[3*(lt*TILE_WIDTH+ct)+1] = image1[ idx+1 ];
			tile[3*(lt*TILE_WIDTH+ct)+2] = image1[ idx+2 ];
		}
		__syncthreads();

		// Aplica o filtro

		int tile_idx = 0;
		int red = 0;
		int green = 0;
		int blue = 0;
		if ( (lt > 0 && lt < TILE_WIDTH - 1) && (ct > 0 && ct < TILE_WIDTH - 1) )
		{

			tile_idx = 3*ELEM( ct, lt, TILE_WIDTH );
			red = tile[ tile_idx+2 ];
			green = tile[ tile_idx+1 ];
			blue = tile[ tile_idx   ];

			int current_color_index = tile_idx+2;

			// Red
			a = red;
			b = tile[current_color_index-3];
			c = tile[current_color_index-3*TILE_WIDTH];
			d = tile[current_color_index+3];
			e = tile[current_color_index+3*TILE_WIDTH];
			
			a1 = sqrt((float)((a-b)*(a-b)));
			a2 = sqrt((float)((a-c)*(a-c)));
			a3 = sqrt((float)((a-d)*(a-d)));
			a4 = sqrt((float)((a-e)*(a-e)));

			float sum = 1 + a1 + a2 + a3 + a4;
			a = (a + a1*b + a2*c + a3*d + a4*e) / sqrt(sum*sum);

			res[ idx+2 ] = (unsigned char) a;

			// Green
			current_color_index = tile_idx+1;
			a = green;
			b = tile[current_color_index-3];
			c = tile[current_color_index-3*TILE_WIDTH];
			d = tile[current_color_index+3];
			e = tile[current_color_index+3*TILE_WIDTH];
			
			a1 = sqrt((float)((a-b)*(a-b)));
			a2 = sqrt((float)((a-c)*(a-c)));
			a3 = sqrt((float)((a-d)*(a-d)));
			a4 = sqrt((float)((a-e)*(a-e)));

			sum = 1 + a1 + a2 + a3 + a4;
			a = (a + a1*b + a2*c + a3*d + a4*e) / sqrt(sum*sum);

			res[ idx+1 ] = (unsigned char) a;

			// Blue
			current_color_index = tile_idx;
			a = blue;
			b = tile[current_color_index-3];
			c = tile[current_color_index-3*TILE_WIDTH];
			d = tile[current_color_index+3];
			e = tile[current_color_index+3*TILE_WIDTH];
			
			a1 = sqrt((float)((a-b)*(a-b)));
			a2 = sqrt((float)((a-c)*(a-c)));
			a3 = sqrt((float)((a-d)*(a-d)));
			a4 = sqrt((float)((a-e)*(a-e)));

			sum = 1 + a1 + a2 + a3 + a4;
			a = (a + a1*b + a2*c + a3*d + a4*e) / sqrt(sum*sum);

			res[ idx ] = (unsigned char) a;
		}
		else
		{
			tile_idx = 3*ELEM( ct, lt, TILE_WIDTH );
			red = tile[ tile_idx+2 ];
			green = tile[ tile_idx+1 ];
			blue = tile[ tile_idx   ];

			res[ idx+2 ] = red;
			res[ idx+1 ] = green;
			res[ idx ] = blue;
		}

		__syncthreads();

	}

}

__host__ double process_in_cuda(char *input_image, 
							  int blSizeX, 
							  int blSizeY,
							  char *output_filename){

	int h_width, h_height;
	unsigned char *h_imagem, *h_imagem_resultado;
	double start_time, gpu_time;

	cout << "\n--------------------------------------------------------------------------------------\n" << endl;
	cout << " \n\t\t\t\t | Filtro utilizando GPU |\n" << endl;

	// Lê a imagem para o buffer h_imagem de entrada
	lerPPM( input_image, &h_imagem, &h_width, &h_height );

	int size = 3*h_width*h_height*sizeof( char );

	// Aloca buffer de resultado no host
	if( ( h_imagem_resultado = (unsigned char *)malloc( size ) ) == NULL )
		erro( "\nErro alocando imagem resultado.\n" );

	// Aloca buffers de entrada para as duas imagens e buffer de saída para imagem de resultado no device
	unsigned char *d_imagem = NULL;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&d_imagem, size ) );
	CUDA_SAFE_CALL( cudaMemcpy( d_imagem, h_imagem, size, cudaMemcpyHostToDevice ) );

	unsigned char *d_res = NULL;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&d_res, size ) );

	// Calcula dimensoes da grid e dos blocos
	dim3 blockSize( blSizeX, blSizeY );
	int numBlocosX = h_width  / blockSize.x + ( h_width  % blockSize.x == 0 ? 0 : 1 );
	int numBlocosY = h_height / blockSize.y + ( h_height % blockSize.y == 0 ? 0 : 1 );
	dim3 gridSize( numBlocosX, numBlocosY, 1 );

	cout << "\nBlocks: (" << blockSize.x << "," << blockSize.y << ")\n";
	cout << "\nGrid: (" << gridSize.x << "," << gridSize.y << ")\n";

	// Chama kernel da GPU
	start_time = get_clock_msec();
	filterGPU<<< gridSize, blockSize >>>( d_imagem, d_res, h_width, h_height  );
	cudaThreadSynchronize();
	gpu_time = get_clock_msec() - start_time;

	// Copia o resultado de volta para o host
	CUDA_SAFE_CALL( cudaMemcpy( h_imagem_resultado, d_res, size, cudaMemcpyDeviceToHost ) );

	// Salva imagem resultado
	salvaPPM( output_filename, h_imagem_resultado, h_width, h_height );

	// Libera memória do device
	CUDA_SAFE_CALL( cudaFree( d_imagem  ) );
	CUDA_SAFE_CALL( cudaFree( d_res     ) );
	
	// Libera memória do host
	free( h_imagem );
	free( h_imagem_resultado );

	return gpu_time;

}


__host__ int main( int argc, char *argv[] ) {

	if( argc < 4 ) 
	{
		erro( "\nSyntaxe: filter -{c,g,s} input_image output_image\n" );
		return 0;
	}

	if ( strcmp(argv[1], CPU) == 0)
	{
		cout << " \nFiltro utilizando CPU\n" << endl;
	}
	else if ( strcmp(argv[1], GPU) == 0)
	{

		double gpu_time = process_in_cuda(argv[2], TILE_WIDTH, TILE_WIDTH, argv[3]);
		// Imprime tempo
		cout << "\n\n\n\t\t\tTempo de execucao da GPU: " << gpu_time << " ms\n" << endl;
		cout << "\n--------------------------------------------------------------------------------------\n" << endl;

	}
	else if ( strcmp(argv[1], SSE) == 0)
	{
		cout << " Filtro utilizando SSE" << endl;
	}
	else
	{
		cout << " Opção inválida" << endl;
		return 0;
	}

	return 0;

}

