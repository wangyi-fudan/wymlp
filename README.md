# Real-Time Intelligence for Every Machine
Tiny fast portable real-time deep neural network for regression and classification with 40 LOC.

Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz Single Thread @ VirtualBox 6.0 

-Ofast -mavx2 -mfma

Speed Measure:	Sample Per Second

|Hidden&Depth|float_training|float_inference|double_training|double_inference|
|----|----|----|----|----|
|16H16L(scalar)|116,849|270,809|109,944|242,912|
|32H16L|**112,504**|197,349|70,236 |142,807|
|64H16L|32,042|58,697|21,635|45,111|
|128H16L|8,715|15,812|4,428|8,904|
|256H16L|2,176|4,388|1,155|2,369|
|512H16L|585|1,201|292|614|

1 minute training on MNIST with 128H2L with single CPU thread.

``

train-images-idx3-ubyte.gz

t10k-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz

t10k-labels-idx1-ubyte.gz

0	error=6.800%	eta=0.300	time=2.374s

1	error=5.010%	eta=0.285	time=4.825s

2	error=4.180%	eta=0.271	time=7.290s

3	error=3.660%	eta=0.257	time=9.754s

4	error=3.430%	eta=0.244	time=12.261s

5	error=3.080%	eta=0.232	time=14.826s

6	error=3.070%	eta=0.221	time=17.355s

7	error=2.830%	eta=0.210	time=19.823s

8	error=2.900%	eta=0.199	time=22.259s

9	error=2.790%	eta=0.189	time=24.711s

10	error=2.620%	eta=0.180	time=27.206s

11	error=2.590%	eta=0.171	time=29.719s

12	error=2.650%	eta=0.162	time=32.191s

13	error=2.560%	eta=0.154	time=34.684s

14	error=2.580%	eta=0.146	time=37.195s

15	error=2.430%	eta=0.139	time=39.693s

16	error=2.400%	eta=0.132	time=42.149s

17	error=2.440%	eta=0.125	time=44.605s

18	error=2.490%	eta=0.119	time=47.115s

19	error=2.440%	eta=0.113	time=49.581s

20	error=2.400%	eta=0.108	time=52.075s

21	error=2.400%	eta=0.102	time=54.539s

22	error=2.490%	eta=0.097	time=57.014s

23	error=2.400%	eta=0.092	time=59.517s

``

Code Example:

```C++
int	main(void){
	float	x[4]={1,2,3,5},	y[1]={2};
	wymlp<float,4,32,16,1,0>	model;	
	for(size_t	i=0;	i<sizeof(model.weight)/sizeof(float);	i++)	model.weight[i]=3.0*rand()/RAND_MAX-1.5;	
	for(unsigned	i=0;	i<1000000;	i++){	
		x[0]+=0.01;	y[0]+=0.1;	//some "new" data
		model.model(x, y, 0.1);	//	training. set eta<0 to predict
	}
	return	0;
}
```
Comments:

0: task=0: regression; task=1: logistic; task=2: softmax

1: eta<0 lead to prediction only.

2: The expected |X[i]|, |Y[i]| should be around 1. Normalize yor input and output first.

3: In practice, it is OK to call model function parallelly with multi-threads, however, they may be slower for small net.

4: The code is portable, however, if Ofast is used on X86, SSE or AVX or even AVX512 will enable very fast code!

5: The default and suggested model is shared hidden-hidden weights. If you want conventional MLP, please replace it with the following lines:
```C++
	#define	woff(i,l)	(l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden)
	type	weight[(input+1)*hidden+(depth-1)*hidden*hidden+output*hidden];
```	


