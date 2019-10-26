# Real-Time Intelligence for Every Machine
Tiny fast portable real-time deep neural network for regression and classification within 50 LOC.

Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz Single Thread @ VirtualBox 6.0 

-Ofast -mavx2 -mfma

Speed Measure:	Sample Per Second

|HiddenUnits,Depth|float_training|float_inference|double_training|double_inference|
|----|----|----|----|----|----|
|4,16|1,217,676||1,073,787 ||
|8,16|408,811||407,376 ||
|16,16|110,779||101,579 ||
|32,16|**90,290**||67,730 ||
|64,16|29,815||18,876 ||
|128,16|8,906||4,334 ||
|256,16|2,029||1,089 ||

Code Example:

```C++
int	main(void){
	float	x[4]={1,2,3,5},	y[1]={2};
	wymlp<float,4,32,16,1,0>	model;	
	model.random(347834);
	for(unsigned	i=0;	i<1000000;	i++){	x[0]++;	model.model(x, y, 0.1);	}
	model.save("model");
	return	0;
}
```
Comments:

0:	task=0: regression; task=1: logistic;	task=2:	softmax

1:	eta<0 lead to prediction only.

2:	The expected |X[i]|, |Y[i]| should be around 1. Normalize yor input and output first.

3:	In practice, it is OK to call model function parallelly with multi-threads, however, they may be slower for small net.

4:	The code is portable, however, if -Ofast is used on X86, autovectorization will make it even faster.

5:	The default and suggested model is shared hidden-hidden weights. If you want conventional MLP, please replace it with the following lines:
```C++
	#define	wymlp_size	((input+1)*hidden+(depth-1)*hidden*hidden+output*hidden)
	unsigned	woff(unsigned	i,	unsigned	l) {	return	l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden;	}
```


