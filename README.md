# wymlp
tiny fast portable deep neural network for regression and classification within 60 LOC 

Example:
```C++
	wymlp<float,4,16,3,1,0>	model;	
	model.ramdom(time(NULL));
	float	x[4]={1,2,3,5},	y[1]={2};
	model.model(x, y, 0.1);	//	train
	model.model(x, y, -1);	//	predict
	model.save("model");
```
Comments:

	0:	task=0: regression; task=1: logistic;	task=2:	softmax

	1:	eta<0 lead to prediction only.

	2:	The expected |X[i]|, |Y[i]| should be around 1. Normalize yor input and output first.

	3:	In practice is OK to call model function parallelly with multi-threads.

	4:	The default and suggested model is shared hidden-hidden weights. If you want conventional MLP, please replace it with the following lines:
  ```C++
	#define	wymlp_size	((input+1)*hidden+(depth-1)*hidden*hidden+output*hidden)
	unsigned	woff(unsigned	i,	unsigned	l) {	return	l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden;	}
  ```
  
	5:	The code is portable, however, if O3 is used on X86, SSE or AVX or even AVX512 will enable very fast code!
