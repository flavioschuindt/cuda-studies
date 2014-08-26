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
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "rf-time.h"
#include "utils.h"
#include "ppm.h"
#include "floatppm.h"
#include "types.h"
extern "C" {
	#include "sse.h"
}


/***************************************************************************************************
	Defines
***************************************************************************************************/
#define CUDA_SAFE_CALL

#define CPU "-c"
#define GPU "-g"
#define GPU_SHARED_MEM "-sm"
#define SSE "-s"
#define TILE_WIDTH 32
#define NUM_THREADS 2 
#define ELEM(i,j,DIMX_) ((i)+(j)*(DIMX_))

/***************************************************************************************************
	Structs
***************************************************************************************************/

struct thread_data thread_data_array[NUM_THREADS];
pthread_t thread_ptr[NUM_THREADS];

/***************************************************************************************************
	Functions
***************************************************************************************************/

using namespace std;

/**************************************************************************************************/

__global__ void filter_gpu_sharedmem( unsigned char *image,
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
			tile[3*(lt*TILE_WIDTH+ct)] = image[ idx ];
			tile[3*(lt*TILE_WIDTH+ct)+1] = image[ idx+1 ];
			tile[3*(lt*TILE_WIDTH+ct)+2] = image[ idx+2 ];
		}
		else
		{
			idx = idx - 2;
			tile[3*(lt*TILE_WIDTH+ct)] = image[ idx ];
			tile[3*(lt*TILE_WIDTH+ct)+1] = image[ idx+1 ];
			tile[3*(lt*TILE_WIDTH+ct)+2] = image[ idx+2 ];
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

__global__ void filter_gpu( unsigned char *image, 
						   unsigned char *res, 
						   int width, 
						   int height ){


	// Calcula o índice da imagem de entrada
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;

	// Coeficientes para o filtro
	float a1, a2, a3, a4;

	// a é o elemento corrente. b, c, d, e são os seus vizinhos 4-conectividade.
	int a, b, c, d, e, k;

	if( (row > 0 && row < height-1) && (col > 0 && col < width-1) )
	{
		int idx = 3*ELEM( col, row, width );
		for (k = 0; k < 3; k++)
		{

			a = image[ idx ];
			b = image[idx-3];
			c = image[idx-3*width];
			d = image[idx+3];
			e = image[idx+3*width];
			
			a1 = sqrt((float)((a-b)*(a-b)));
			a2 = sqrt((float)((a-c)*(a-c)));
			a3 = sqrt((float)((a-d)*(a-d)));
			a4 = sqrt((float)((a-e)*(a-e)));

			float sum = 1 + a1 + a2 + a3 + a4;
			a = (a + a1*b + a2*c + a3*d + a4*e) / sqrt(sum*sum);

			res[ idx++ ] = (unsigned char) a;

		}
	}

}

__host__ double process_in_cuda(char *input_image, 
							  int blSizeX, 
							  int blSizeY,
							  char *output_filename,
							  int is_sharedmem){

	int h_width, h_height;
	unsigned char *h_imagem, *h_imagem_resultado;
	double start_time, gpu_time;

	// Lê a imagem para o buffer h_imagem de entrada
	lerPPM( input_image, &h_imagem, &h_width, &h_height );

	int size = 3*h_width*h_height*sizeof( char );

	// Aloca buffer de resultado no host
	if( ( h_imagem_resultado = (unsigned char *)malloc( size ) ) == NULL )
		erro( "\n[CUDA] Erro alocando imagem resultado.\n" );

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

	cout << "\n[CUDA] Blocks: (" << blockSize.x << "," << blockSize.y << ")\n";
	cout << "\n[CUDA] Grid: (" << gridSize.x << "," << gridSize.y << ")\n";

	if (is_sharedmem)
	{
		// Chama kernel da GPU
		start_time = get_clock_msec();
		filter_gpu_sharedmem<<< gridSize, blockSize >>>( d_imagem, d_res, h_width, h_height  );
		cudaThreadSynchronize();
		gpu_time = get_clock_msec() - start_time;
	}
	else
	{
		start_time = get_clock_msec();
		filter_gpu<<< gridSize, blockSize >>>( d_imagem, d_res, h_width, h_height  );
		cudaThreadSynchronize();
		gpu_time = get_clock_msec() - start_time;
	}

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

void* sse( void *threadarg ) {

	return do_sse(threadarg, NUM_THREADS);

}

__host__ double process_in_cpu(char *input_image, 
							  char *output_filename,
							  int is_sse){

	int width, height, size;
	unsigned char *imagem, *imagem_resultado;
	float *imagem_sse, *imagem_sse_resultado;
	double start_time, cpu_time;
	int i, j, k, idx;
	float a1, a2, a3, a4; // Coeficientes para o filtro
	int a, b, c, d, e; // a é o elemento corrente. b, c, d, e são os seus vizinhos 4-conectividade.

	if (is_sse)
	{

		readPPMinFloat( input_image, &imagem_sse, &width, &height );

		size = 3*width*height*sizeof( float );

		posix_memalign( (void **)&imagem_sse_resultado, 16, size );

		if( imagem_sse_resultado == NULL ) {
			printf( "[SSE] Erro alocando imagem resultado.\n" );
			exit( 1 );
		}

		start_time = get_clock_msec();

		for( int i = 0 ; i < NUM_THREADS ; i++ ) {
			
			thread_data_array[i].thread_id = i;
			thread_data_array[i].im1 = imagem_sse;
			thread_data_array[i].res = imagem_sse_resultado;
			thread_data_array[i].w = width;
			thread_data_array[i].h = height;

			pthread_create( &thread_ptr[i], NULL, sse, (void *)&thread_data_array[i] );
			
		}

		/* Wait for every thread to complete  */
		for( int i = 0 ; i < NUM_THREADS ; i++ ) {
			pthread_join(thread_ptr[i], NULL);
		}
		cpu_time = get_clock_msec() - start_time;

		// Salva imagem resultado
		savePPMfromFloat( output_filename, imagem_sse_resultado, width, height );
		free( imagem_sse );
		free( imagem_sse_resultado );
	}
	else
	{
		// Lê a imagem para o buffer h_imagem de entrada
		lerPPM( input_image, &imagem, &width, &height );

		int size = 3*width*height*sizeof( char );

		// Aloca buffer de resultado
		if( ( imagem_resultado = (unsigned char *)malloc( size ) ) == NULL )
			erro( "\n[CPU] Erro alocando imagem resultado.\n" );


		start_time = get_clock_msec();
		for (i = 1; i < height-1; i++)
		{
			for (j = 1; j < width-1; j++)
			{

				idx = 3*ELEM(j, i, width);
				for (k = 0; k < 3; k++)
				{

					a = imagem[ idx ];
					b = imagem[idx-3];
					c = imagem[idx-3*width];
					d = imagem[idx+3];
					e = imagem[idx+3*width];
					
					a1 = sqrt((float)((a-b)*(a-b)));
					a2 = sqrt((float)((a-c)*(a-c)));
					a3 = sqrt((float)((a-d)*(a-d)));
					a4 = sqrt((float)((a-e)*(a-e)));

					float sum = 1 + a1 + a2 + a3 + a4;
					a = (a + a1*b + a2*c + a3*d + a4*e) / sqrt(sum*sum);

					imagem_resultado[ idx++ ] = (unsigned char) a;

				}
			}
		}
		cpu_time = get_clock_msec() - start_time;

		// Salva imagem resultado
		salvaPPM( output_filename, imagem_resultado, width, height );
		free( imagem );
		free( imagem_resultado );
	}
	

	return cpu_time;

}

__host__ int main( int argc, char *argv[] ) {

	if( argc < 4 ) 
	{
		erro( "\nSyntaxe: filter -{c,g,s,sm} input_image output_image\n" );
		return 0;
	}

	if ( strcmp(argv[1], CPU) == 0)
	{
		cout << "\n--------------------------------------------------------------------------------------\n" << endl;
		cout << " \n\t\t\t\t[CPU] | Filtro utilizando CPU |\n" << endl;

		double cpu_time = process_in_cpu(argv[2], argv[3], 0);

		cout << "\n\n\n\t\t\t[CPU] Tempo de execucao da CPU (sem SSE): " << cpu_time << " ms\n" << endl;
		cout << "\n--------------------------------------------------------------------------------------\n" << endl;
	}
	else if ( strcmp(argv[1], GPU_SHARED_MEM) == 0)
	{

		cout << "\n--------------------------------------------------------------------------------------\n" << endl;
		cout << " \n\t\t\t\t[CUDA] | Filtro utilizando GPU com Shared Memory |\n" << endl;
		
		double gpu_time = process_in_cuda(argv[2], TILE_WIDTH, TILE_WIDTH, argv[3], 1);
		
		cout << "\n\n\n\t\t\t[CUDA] Tempo de execucao da GPU com Shared Memory: " << gpu_time << " ms\n" << endl;
		cout << "\n--------------------------------------------------------------------------------------\n" << endl;

	}
	else if ( strcmp(argv[1], GPU) == 0)
	{

		cout << "\n--------------------------------------------------------------------------------------\n" << endl;
		cout << " \n\t\t\t\t[CUDA] | Filtro utilizando GPU |\n" << endl;
		
		double gpu_time = process_in_cuda(argv[2], TILE_WIDTH, TILE_WIDTH, argv[3], 0);
		
		cout << "\n\n\n\t\t\t[CUDA] Tempo de execucao da GPU: " << gpu_time << " ms\n" << endl;
		cout << "\n--------------------------------------------------------------------------------------\n" << endl;

	}
	else if ( strcmp(argv[1], SSE) == 0)
	{
		cout << "\n--------------------------------------------------------------------------------------\n" << endl;
		cout << " \n\t\t\t\t[MERGE SSE] | Filtro utilizando CPU SSE |\n" << endl;
		
		double cpu_time = process_in_cpu(argv[2], argv[3], 1);
		
		cout << "\n\n\n\t\t\t[MERGE SSE] Tempo de execucao da CPU (com SSE): " << cpu_time << " ms\n" << endl;
		cout << "\n--------------------------------------------------------------------------------------\n" << endl;
	}
	else
	{
		cout << " Opção inválida" << endl;
		return 0;
	}

	return 0;

}

