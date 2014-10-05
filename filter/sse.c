#include <xmmintrin.h>
#include <math.h>
#include "sse.h"
#include "types.h"

void* do_sse(void *threadarg, int num_threads)
{
	__m128 current_element, left, top, right, bottom, numerator, denominator, a1, a2, a3, a4, aux;
	int id, w, h, limit, idx, i, line_offset, width_limit;
	float *imagem, *res;
	float rgb[4];
	float numerator_values[4];
	float denominator_values[4];
	float num, dem;

	struct thread_data *thread_pointer_data;
	thread_pointer_data = (struct thread_data *)threadarg;

	id = thread_pointer_data->thread_id;
	imagem = (float *)thread_pointer_data->im1;
	res = thread_pointer_data->res;
	w = thread_pointer_data->w;
	h = thread_pointer_data->h;

	limit = 3*w*h / num_threads;
	numerator = _mm_setzero_ps(); // (0, 0, 0, 0)
	denominator = _mm_setzero_ps(); // (0, 0, 0, 0)

	width_limit = 3*w;
	for( idx = id*limit ; idx < (id+1)*limit ; idx+=3 ) 
	{
		line_offset =  idx / width_limit;
		if (
			idx >= width_limit &&
			idx < width_limit*(h-1) &&
			idx % (width_limit) != 0 &&
			idx != (line_offset*(width_limit) + (width_limit-3))
			)
		{
			// Current element
			rgb[0] = imagem[ idx ]; // Red
			rgb[1] = imagem[ idx+1 ]; // Green
			rgb[2] = imagem[ idx+2 ]; // Blue
			rgb[3] = 0;
			current_element = _mm_load_ps(&rgb[0]);

			// Left
			rgb[0] = imagem[ idx-3 ]; // Red
			rgb[1] = imagem[ idx-2 ]; // Green
			rgb[2] = imagem[ idx-1 ]; // Blue
			rgb[3] = 0;
			left = _mm_load_ps(&rgb[0]);

			// Right
			rgb[0] = imagem[ idx+3 ]; // Red
			rgb[1] = imagem[ idx+4 ]; // Green
			rgb[2] = imagem[ idx+5 ]; // Blue
			rgb[3] = 0;
			right = _mm_load_ps(&rgb[0]);

			// Top
			rgb[0] = imagem[ idx-3*w ]; // Red
			rgb[1] = imagem[ idx-3*w+1 ]; // Green
			rgb[2] = imagem[ idx-3*w+2 ]; // Blue
			rgb[3] = 0;
			top = _mm_load_ps(&rgb[0]);

			// Bottom
			rgb[0] = imagem[ idx+3*w ]; // Red
			rgb[1] = imagem[ idx+3*w+1 ]; // Green
			rgb[2] = imagem[ idx+3*w+2 ]; // Blue
			rgb[3] = 0;
			bottom = _mm_load_ps(&rgb[0]);

			a1 = _mm_sqrt_ps(_mm_mul_ps(_mm_sub_ps(current_element, left), _mm_sub_ps(current_element, left)));
			a2 = _mm_sqrt_ps(_mm_mul_ps(_mm_sub_ps(current_element, top), _mm_sub_ps(current_element, top)));
			a3 = _mm_sqrt_ps(_mm_mul_ps(_mm_sub_ps(current_element, right), _mm_sub_ps(current_element, right)));
			a4 = _mm_sqrt_ps(_mm_mul_ps(_mm_sub_ps(current_element, bottom), _mm_sub_ps(current_element, bottom)));

			rgb[0] = 1;
			rgb[1] = 1;
			rgb[2] = 1;
			rgb[3] = 0;

			aux = _mm_load_ps(&rgb[0]);

			numerator = _mm_add_ps(current_element, 
								  _mm_add_ps
								  (
								  	_mm_add_ps
								  		(
								  		_mm_mul_ps(a1, left), _mm_mul_ps(a2, top)
								  		),
									_mm_add_ps
										(
										_mm_mul_ps(a3, right), _mm_mul_ps(a4, bottom)
										)
								   )
								); //a + a1*b + a2*c + a3*d + a4*e

			__m128 sum = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(a1, a2), a3), a4), aux);  //1 + a1 + a2 + a3 + a4
			denominator = _mm_sqrt_ps(_mm_mul_ps(sum, sum)); //sqrt(sum*sum)

			_mm_store_ps (numerator_values, numerator);
			_mm_store_ps (denominator_values, denominator);

			for (i=0; i<3; i++)
			{
	    		res[ idx+i ] = numerator_values[i] / denominator_values[i];
	    	}
		}
	}
	
}