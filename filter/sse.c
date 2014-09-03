#include <xmmintrin.h>
#include <math.h>
#include "sse.h"
#include "types.h"

void* do_sse(void *threadarg, int num_threads)
{
	int id, idx, limit;
	__m128 elements_vector, current_element_vector, s, aux;
	float *imagem, *res;
	int w, h;
	float elements[4];
	float current_element[4];
	float numerator_sum, denominator_sum;

	s = _mm_setzero_ps(); // (0, 0, 0, 0)

	struct thread_data *thread_pointer_data;
	thread_pointer_data = (struct thread_data *)threadarg;

	id = thread_pointer_data->thread_id;
	imagem = thread_pointer_data->im1;
	res = thread_pointer_data->res;
	w = thread_pointer_data->w;
	h = thread_pointer_data->h;

	limit = 3*w*h / num_threads;
	
	for( idx = id*limit ; idx < (id+1)*limit ; idx++ ) {

			elements[0] = imagem[idx-3];
			elements[1] = imagem[idx-3*w];
			elements[2] = imagem[idx+3];
			elements[3] = imagem[idx+3*w];

			current_element[0] = imagem[idx];
			current_element[1] = imagem[idx];
			current_element[2] = imagem[idx];
			current_element[3] = imagem[idx];

			elements_vector = _mm_load_ps(&elements[0]);
			current_element_vector = _mm_load_ps(&current_element[0]);

			s = _mm_sub_ps(current_element_vector, elements_vector);
			s = _mm_mul_ps(s, s);

			aux = _mm_hadd_ps(s, s);      // sum horizontally
			aux = _mm_hadd_ps(aux, aux);  // (NB: need to do this twice to sum all 4 elements)*/
			/*aux = _mm_add_ps(s, _mm_movehl_ps(s, s));
			__m128 sumxx = _mm_add_ss(aux, _mm_shuffle_ps(aux, aux, 1));*/
			_mm_store_ss(&denominator_sum, aux); 
			denominator_sum += 1.0;

			aux = _mm_mul_ps(s, elements_vector);
			aux = _mm_hadd_ps(aux, aux);   // sum horizontally
			aux = _mm_hadd_ps(aux, aux);  // (NB: need to do this twice to sum all 4 elements)
			/*aux = _mm_add_ps(aux, _mm_movehl_ps(aux, aux));
			sumxx = _mm_add_ss(aux, _mm_shuffle_ps(aux, aux, 1));*/
			_mm_store_ss(&numerator_sum, aux);
			numerator_sum += imagem[ idx ]; 
			
			imagem[ idx ] = numerator_sum / sqrt(denominator_sum * denominator_sum);
	}
}