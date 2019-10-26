# Real-Time Intelligence for Every Machine
Tiny fast portable real-time deep neural network for regression and classification with 40 LOC.

## Benchmark
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

## MNIST
1 minute training on MNIST with 64H3L with single CPU thread.

```
0	error=8.000%	eta=0.300	time=1.260s
1	error=5.920%	eta=0.285	time=2.537s
2	error=4.920%	eta=0.271	time=3.813s
3	error=4.160%	eta=0.257	time=5.078s
4	error=3.970%	eta=0.244	time=6.343s
5	error=3.480%	eta=0.232	time=7.603s
6	error=3.170%	eta=0.221	time=8.875s
7	error=3.080%	eta=0.210	time=10.138s
8	error=3.090%	eta=0.199	time=11.411s
9	error=3.160%	eta=0.189	time=12.684s
10	error=3.100%	eta=0.180	time=13.950s
11	error=2.780%	eta=0.171	time=15.215s
12	error=3.030%	eta=0.162	time=16.483s
13	error=2.710%	eta=0.154	time=17.766s
14	error=2.970%	eta=0.146	time=19.037s
15	error=2.920%	eta=0.139	time=20.298s
16	error=2.560%	eta=0.132	time=21.571s
17	error=2.530%	eta=0.125	time=22.842s
18	error=2.610%	eta=0.119	time=24.108s
19	error=2.530%	eta=0.113	time=25.381s
20	error=2.700%	eta=0.108	time=26.650s
21	error=2.550%	eta=0.102	time=27.917s
22	error=2.510%	eta=0.097	time=29.177s
23	error=2.340%	eta=0.092	time=30.445s
24	error=2.460%	eta=0.088	time=31.715s
25	error=2.390%	eta=0.083	time=32.968s
26	error=2.380%	eta=0.079	time=34.229s
27	error=2.310%	eta=0.075	time=35.509s
28	error=2.370%	eta=0.071	time=36.794s
29	error=2.560%	eta=0.068	time=38.071s
30	error=2.390%	eta=0.064	time=39.375s
31	error=2.460%	eta=0.061	time=40.652s
32	error=2.330%	eta=0.058	time=41.931s
33	error=2.270%	eta=0.055	time=43.197s
34	error=2.270%	eta=0.052	time=44.477s
35	error=2.380%	eta=0.050	time=45.745s
36	error=2.290%	eta=0.047	time=47.026s
37	error=2.220%	eta=0.045	time=48.300s
38	error=2.290%	eta=0.043	time=49.579s
39	error=2.210%	eta=0.041	time=50.857s
40	error=2.240%	eta=0.039	time=52.128s
41	error=2.180%	eta=0.037	time=53.415s
42	error=2.260%	eta=0.035	time=54.681s
43	error=2.290%	eta=0.033	time=55.955s
44	error=2.180%	eta=0.031	time=57.225s
45	error=2.210%	eta=0.030	time=58.498s
46	error=2.130%	eta=0.028	time=59.769s
```
### Approximate Time Comparision

wymlp runs on my destop mentioned above while other benchmark comes from https://github.com/attractivechaos/kann which runs on Xeno E5-2697 CPUs at 2.7GHz. 64H1L setting.

|Algorithm|time|
|----|----|
|wymlp(different machinine)|15.387s|
|KANN+SSE|31.2s|
|KANN+BLAS|18.8s|
|Theano+Keras|33.2s|
|TensorFlow|33.4s|
|Tiny-dnn|2m18s|
|Tiny-dnn+AVX|1m33s|

## Code Example:

```C++
int	main(void){
	float	x[4]={1,2,3,5},	y[1]={2};
	wymlp<float,4,32,16,1,0>	model;	
	for(size_t	i=0;	i<sizeof(model.weight)/sizeof(float);	i++)	model.weight[i]=3.0*rand()/RAND_MAX-1.5;	
	for(unsigned	i=0;	i<1000000;	i++){	
		x[0]+=0.01;	y[0]+=0.1;	//some "new" data
		model.model(x, y, 0.1,	0.5,	wygrand());	//	training. set eta<0 to predict
	}
	return	0;
}
```
## Comments:

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


