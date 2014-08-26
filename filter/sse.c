#include <xmmintrin.h>
#include <math.h>
#include "sse.h"
#include "types.h"

void* do_sse(void *threadarg, int num_threads)
{
	/*int id, idx, limit;
	__m128 m1, s;
	float *imagem, *res;
	int w, h;

	s = _mm_setzero_ps(); // (0, 0, 0, 0)

	struct thread_data *thread_pointer_data;
	thread_pointer_data = (struct thread_data *)threadarg;

	id = thread_pointer_data->thread_id;
	imagem = thread_pointer_data->im1;
	res = thread_pointer_data->res;
	w = thread_pointer_data->w;
	h = thread_pointer_data->h;

	limit = (3*w*h / num_threads) - 4;
	int horizontal_upper_border_limit = 3*w;
	int horizontal_bottom_border_limit = 3*w*h - w;

	// Coeficientes para o filtro
	float a1, a2, a3, a4;

	// a é o elemento corrente. b, c, d, e são os seus vizinhos 4-conectividade.
	float a, b, c, d, e;
	
	for( idx = id*limit ; idx < (id+1)*limit ; idx+=4 ) {
		
		if ( (idx >= horizontal_upper_border_limit) && (idx < horizontal_bottom_border_limit) )
		{

			m1 = _mm_load_ps(&imagem[idx]);

			b = imagem[idx-3];
			c = imagem[idx-3*w];
			d = imagem[idx+3];
			e = imagem[idx+3*w];
			
			a1 = sqrt(((a-b)*(a-b)));
			a2 = sqrt(((a-c)*(a-c)));
			a3 = sqrt(((a-d)*(a-d)));
			a4 = sqrt(((a-e)*(a-e)));

			float sum = 1 + a1 + a2 + a3 + a4;
			a = (a + a1*b + a2*c + a3*d + a4*e) / sqrt(sum*sum);

			res[ idx++ ] = (unsigned char) a;

			s = _mm_mul_ps(_mm_add_ps(m1,m2), factor);
			_mm_store_ps(&res[idx], s);
		}
	}*/
}