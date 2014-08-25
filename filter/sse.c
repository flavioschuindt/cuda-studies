#include <xmmintrin.h>
#include "sse.h"

void* do_sse(void *threadarg)
{
	int id, idx, limit;
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

	limit = (3*w*h / NUM_THREADS) - 4;
	int horizontal_upper_border_limit = 3*w;
	int horizontal_bottom_border_limit = 3*w*h - w;
	
	for( int idx = id*limit ; idx < (id+1)*limit ; idx+=4 ) {
		
		if ( (idx >= horizontal_upper_border_limit) && (idx < horizontal_bottom_border_limit) )
		{

			m1 = _mm_load_ps(&imagem[idx]);

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

			s = _mm_mul_ps(_mm_add_ps(m1,m2), factor);
			_mm_store_ps(&res[idx], s);
		}
	}
}