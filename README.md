# Real-Time Intelligence for Every Machine
Tiny fast portable real-time deep neural network for regression and classification within 50 LOC.

## Benchmark
Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz Single Thread @ VirtualBox 6.0 

-Ofast -mavx2 -mfma

Speed Measure:	Sample Per Second

|Hidden&Depth|float_training|float_inference|double_training|double_inference|
|----|----|----|----|----|
|16H16L(scalar)|141,958|397,894|140,921|368,760|
|32H16L|155,054|297,814|96,204 |208,723|
|64H16L|47,253|95,647|24,794|52,018|
|128H16L|11,469|25,124|4,950|11,696|
|256H16L|2,356|5,502|1,260|2,725|
|512H16L|610|1,246|312|648|

## Code Example:

```C++
int	main(void){
	float	x[4]={1,2,3,5},	y[1]={2};
	vector<float>	weight(wymlp<float,12,32,4,1,0>(NULL,NULL,NULL,0,0,-1));	//set dropout<0 to return size
	for(size_t	i=0;	i<weight.size();	i++)	weight[i]=3.0*rand()/RAND_MAX-1.5;	
	for(unsigned	i=0;	i<1000000;	i++){	
		x[0]+=0.01;	y[0]+=0.1;	//some "new" data
		wymlp<float,12,32,4,1,0>(weight.data(),	x, y, 0.1,	wygrand(),	0.5);	//	training. set eta>0 to train
		wymlp<float,12,32,4,1,0>(weight.data(),	x, y, -1,	wygrand(),	0.5);	//	training. set eta<0 to predict
	}
	return	0;
}
```
Comments:

0: loss=0: regression; loss=1: logistic; loss=2: softmax

1: dropout<0 lead to size() function

2: eta<0 lead to prediction only.

3: The expected |X[i]|, |Y[i]| should be around 1. Normalize yor input and output first.

4: In practice, it is OK to call model function parallelly with multi-threads, however, they may be slower for small net.

5: The code is portable, however, if Ofast is used on X86, SSE or AVX or even AVX512 will enable very fast code!

6: The default and suggested model is shared hidden-hidden weights. If you want vanilla MLP, define VanillaMLP


## MNIST test error of 128H2L with single CPU thread:

```
0	error=8.240%	eta=0.300	time=2.058s
1	error=6.720%	eta=0.297	time=4.081s
2	error=5.660%	eta=0.294	time=6.112s
3	error=5.200%	eta=0.291	time=8.137s
4	error=4.330%	eta=0.288	time=10.154s
5	error=4.050%	eta=0.285	time=12.184s
6	error=3.740%	eta=0.282	time=14.206s
7	error=3.360%	eta=0.280	time=16.220s
8	error=3.550%	eta=0.277	time=18.274s
9	error=3.170%	eta=0.274	time=20.350s
10	error=2.980%	eta=0.271	time=22.373s
11	error=3.080%	eta=0.269	time=24.452s
12	error=2.670%	eta=0.266	time=26.561s
13	error=2.880%	eta=0.263	time=28.622s
14	error=2.430%	eta=0.261	time=30.712s
15	error=2.660%	eta=0.258	time=32.788s
16	error=2.330%	eta=0.255	time=34.841s
17	error=2.420%	eta=0.253	time=36.871s
18	error=2.470%	eta=0.250	time=38.931s
19	error=2.160%	eta=0.248	time=40.973s
20	error=2.180%	eta=0.245	time=43.015s
21	error=2.220%	eta=0.243	time=45.082s
22	error=2.240%	eta=0.240	time=47.163s
23	error=2.310%	eta=0.238	time=49.222s
24	error=2.100%	eta=0.236	time=51.292s
25	error=2.110%	eta=0.233	time=53.365s
26	error=2.050%	eta=0.231	time=55.427s
27	error=2.060%	eta=0.229	time=57.483s
28	error=2.160%	eta=0.226	time=59.564s
29	error=2.110%	eta=0.224	time=61.624s
30	error=2.060%	eta=0.222	time=63.675s
31	error=2.010%	eta=0.220	time=65.721s
32	error=2.000%	eta=0.217	time=67.772s
33	error=1.960%	eta=0.215	time=69.810s
34	error=1.900%	eta=0.213	time=71.860s
35	error=2.040%	eta=0.211	time=73.899s
36	error=1.960%	eta=0.209	time=75.955s
37	error=1.880%	eta=0.207	time=78.036s
38	error=1.980%	eta=0.205	time=80.074s
39	error=1.840%	eta=0.203	time=82.146s
40	error=1.910%	eta=0.201	time=84.217s
41	error=1.900%	eta=0.199	time=86.276s
42	error=1.870%	eta=0.197	time=88.361s
43	error=1.810%	eta=0.195	time=90.478s
44	error=1.720%	eta=0.193	time=92.610s
45	error=1.860%	eta=0.191	time=94.665s
46	error=1.820%	eta=0.189	time=96.767s
47	error=1.760%	eta=0.187	time=98.826s
48	error=1.870%	eta=0.185	time=100.898s
49	error=2.000%	eta=0.183	time=102.945s
50	error=1.680%	eta=0.182	time=104.979s
51	error=1.750%	eta=0.180	time=107.059s
52	error=1.710%	eta=0.178	time=109.127s
53	error=1.790%	eta=0.176	time=111.144s
54	error=1.870%	eta=0.174	time=113.179s
55	error=1.720%	eta=0.173	time=115.232s
56	error=1.770%	eta=0.171	time=117.281s
57	error=1.690%	eta=0.169	time=119.326s
58	error=1.770%	eta=0.167	time=121.387s
59	error=1.740%	eta=0.166	time=123.436s
60	error=1.740%	eta=0.164	time=125.487s
61	error=1.870%	eta=0.163	time=127.522s
62	error=1.790%	eta=0.161	time=129.578s
63	error=1.730%	eta=0.159	time=131.622s
64	error=1.760%	eta=0.158	time=133.669s
65	error=1.790%	eta=0.156	time=135.743s
66	error=1.640%	eta=0.155	time=137.811s
67	error=1.640%	eta=0.153	time=139.856s
68	error=1.610%	eta=0.151	time=141.894s
69	error=1.600%	eta=0.150	time=143.962s
70	error=1.680%	eta=0.148	time=145.992s
71	error=1.590%	eta=0.147	time=148.026s

```


