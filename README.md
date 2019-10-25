# Simple is Best. Intelligence Everywhere!
Tiny fast portable deep neural network for regression and classification within 60 LOC. Single thread is faster than multiple threads due to memory bound and CPU coherence.

Example:
```C++
	wymlp<float,4,16,3,1,0>	model;	
	model.ramdom(time(NULL));
	float	x[4]={1,2,3,5},	y[1]={2};
	model.model(x, y, 0.1);	//	train
	model.model(x, y, -1);	//	predict
	model.save("model");
```

0:	task=0: regression; task=1: logistic;	task=2:	softmax

1:	eta<0 lead to prediction only.

2:	The expected |X[i]|, |Y[i]| should be around 1. Normalize yor input and output first.

3:	In practice, it is OK to call model function parallelly with multi-threads, however, they may be slower for small net.

4:	The code is portable, however, if O3 is used on X86, SSE or AVX or even AVX512 will enable very fast code!

5:	The default and suggested model is shared hidden-hidden weights. If you want conventional MLP, please replace it with the following lines:
```C++
	#define	wymlp_size	((input+1)*hidden+(depth-1)*hidden*hidden+output*hidden)
	unsigned	woff(unsigned	i,	unsigned	l) {	return	l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden;	}
```

Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz Single Thread @ VirtualBox 6.0

|HiddenUnits|	float/GFLOPS	|double/GFLOPS|long double/GFLOPS|
|----|----|----|----|
|4|	2.3| 	2.1 |0.78|
|8|	3.0|	2.8 |0.66|
|16|	3.3| 	3.1 |0.71|
|32|	10.5| 	6.6 |0.76|
|64|	13.7| 	7.8 |0.83|
|128|	15.1| 	6.3 |0.84|
|256|	12.2| 	6.7 |0.84|

Although GFLOPS is not large, the "FPS" (100,000) can be very large for small deep network(32 hidden units, 16 layers). It is OK for real time processing each WAV frame.
