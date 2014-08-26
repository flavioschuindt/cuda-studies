//----------------------------------------------------------
// Prog: types.h
// Goal: General data structures
// Date: August 26 2014
// By  : Flávio Schuindt
//----------------------------------------------------------
#ifndef TYPES_H_
#define TYPES_H_

struct thread_data {
		unsigned int thread_id;
		float *im1;
		float *res;
		int w;
		int h;
	};

#endif