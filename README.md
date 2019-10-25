# Simple is Best. Intelligence Everywhere!
Tiny fast portable deep neural network for regression and classification within 50 LOC. It reaches 100k FPS trainning on X86 which is ideal for real time smart information processing.

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

TSPS=Training Sample Per Second

|HiddenUnits,Depth|float/TSPS|double/TSPS|autovectorization|
|----|----|----|----|
|4,16|	1,217,676| 	1,073,787 |scalar|
|8,16|	408,811|	407,376 |scalar|
|16,16|	110,779| 	101,579 |scalar
|32,16|	**90,290**| 	67,730 |vectorized|
|64,16|	29,815| 	18,876 |vectorized|
|128,16|	8,906| 	4,334 |vectorized|
|256,16|	2,029| 	1,089 |vectorized|
