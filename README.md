# Real-Time Intelligence for Every Machine
Tiny fast portable real-time deep neural network for regression and classification with 40 LOC.

## Benchmark
Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz Single Thread @ VirtualBox 6.0 

-Ofast -mavx2 -mfma

Speed Measure:	Sample Per Second

|Hidden&Depth|float_training|float_inference|double_training|double_inference|
|----|----|----|----|----|
|16H16L(scalar)|141,958|397,894|109,944|242,912|
|32H16L|**155,054**|297,814|70,236 |142,807|
|64H16L|32,042|58,697|21,635|45,111|
|128H16L|8,715|15,812|4,428|8,904|
|256H16L|2,176|4,388|1,155|2,369|
|512H16L|585|1,201|292|614|

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

## MNIST test error of 128H2L with single CPU thread:

```
0	error=8.140%	eta=0.300	time=2.217s
1	error=6.930%	eta=0.297	time=4.399s
2	error=5.810%	eta=0.294	time=6.581s
3	error=4.940%	eta=0.291	time=8.774s
4	error=4.370%	eta=0.288	time=10.971s
5	error=3.960%	eta=0.285	time=13.142s
6	error=3.620%	eta=0.282	time=15.329s
7	error=3.210%	eta=0.280	time=17.516s
8	error=3.270%	eta=0.277	time=19.708s
9	error=2.980%	eta=0.274	time=21.895s
10	error=3.080%	eta=0.271	time=24.111s
11	error=2.980%	eta=0.269	time=26.308s
12	error=3.050%	eta=0.266	time=28.525s
13	error=2.680%	eta=0.263	time=30.740s
14	error=2.530%	eta=0.261	time=32.980s
15	error=2.560%	eta=0.258	time=35.187s
16	error=2.730%	eta=0.255	time=37.402s
17	error=2.510%	eta=0.253	time=39.619s
18	error=2.520%	eta=0.250	time=41.829s
19	error=2.260%	eta=0.248	time=44.051s
20	error=2.560%	eta=0.245	time=46.265s
21	error=2.410%	eta=0.243	time=48.480s
22	error=2.440%	eta=0.240	time=50.701s
23	error=2.170%	eta=0.238	time=52.914s
24	error=2.130%	eta=0.236	time=55.138s
25	error=2.130%	eta=0.233	time=57.347s
26	error=2.160%	eta=0.231	time=59.574s
27	error=2.070%	eta=0.229	time=61.810s
28	error=2.190%	eta=0.226	time=64.027s
29	error=2.170%	eta=0.224	time=66.257s
30	error=1.990%	eta=0.222	time=68.463s
31	error=2.100%	eta=0.220	time=70.680s
32	error=2.040%	eta=0.217	time=72.914s
33	error=2.130%	eta=0.215	time=75.131s
34	error=1.770%	eta=0.213	time=77.339s
35	error=1.880%	eta=0.211	time=79.560s
36	error=1.980%	eta=0.209	time=81.769s
37	error=1.920%	eta=0.207	time=83.973s
38	error=1.890%	eta=0.205	time=86.198s
39	error=1.910%	eta=0.203	time=88.425s
40	error=1.960%	eta=0.201	time=90.637s
41	error=1.850%	eta=0.199	time=92.865s
42	error=1.860%	eta=0.197	time=95.049s
43	error=1.940%	eta=0.195	time=97.272s
44	error=1.840%	eta=0.193	time=99.485s
45	error=1.720%	eta=0.191	time=101.693s
46	error=1.670%	eta=0.189	time=103.909s
47	error=1.890%	eta=0.187	time=106.132s
48	error=1.880%	eta=0.185	time=108.342s
49	error=1.810%	eta=0.183	time=110.560s
50	error=1.870%	eta=0.182	time=112.779s
51	error=1.740%	eta=0.180	time=115.031s
52	error=1.730%	eta=0.178	time=117.260s
53	error=1.790%	eta=0.176	time=119.456s
54	error=1.710%	eta=0.174	time=121.668s
55	error=1.710%	eta=0.173	time=123.905s
56	error=1.700%	eta=0.171	time=126.095s
57	error=1.910%	eta=0.169	time=128.339s
58	error=1.630%	eta=0.167	time=130.581s
59	error=1.850%	eta=0.166	time=132.809s
60	error=1.730%	eta=0.164	time=135.017s
61	error=1.720%	eta=0.163	time=137.222s
62	error=1.680%	eta=0.161	time=139.456s
63	error=1.660%	eta=0.159	time=141.704s
64	error=1.710%	eta=0.158	time=143.941s
65	error=1.690%	eta=0.156	time=146.235s
66	error=1.630%	eta=0.155	time=148.506s
67	error=1.620%	eta=0.153	time=150.730s
68	error=1.660%	eta=0.151	time=152.976s
69	error=1.560%	eta=0.150	time=155.234s
70	error=1.600%	eta=0.148	time=157.506s
71	error=1.620%	eta=0.147	time=159.757s
72	error=1.640%	eta=0.145	time=162.007s
73	error=1.640%	eta=0.144	time=164.245s
74	error=1.580%	eta=0.143	time=166.508s
75	error=1.650%	eta=0.141	time=168.855s
76	error=1.710%	eta=0.140	time=171.132s
77	error=1.590%	eta=0.138	time=173.359s
78	error=1.510%	eta=0.137	time=175.575s
79	error=1.720%	eta=0.136	time=177.805s
80	error=1.520%	eta=0.134	time=180.073s
81	error=1.620%	eta=0.133	time=182.366s
82	error=1.640%	eta=0.132	time=184.668s
83	error=1.530%	eta=0.130	time=186.891s
84	error=1.630%	eta=0.129	time=189.115s
85	error=1.540%	eta=0.128	time=191.352s
86	error=1.510%	eta=0.126	time=193.619s
87	error=1.420%	eta=0.125	time=195.878s
88	error=1.560%	eta=0.124	time=198.121s
89	error=1.580%	eta=0.123	time=200.377s
90	error=1.560%	eta=0.121	time=202.645s
91	error=1.600%	eta=0.120	time=204.924s
92	error=1.630%	eta=0.119	time=207.181s
93	error=1.510%	eta=0.118	time=209.448s
94	error=1.600%	eta=0.117	time=211.724s
95	error=1.720%	eta=0.115	time=214.006s
96	error=1.580%	eta=0.114	time=216.297s
97	error=1.580%	eta=0.113	time=218.568s
98	error=1.510%	eta=0.112	time=220.831s
99	error=1.630%	eta=0.111	time=223.121s
100	error=1.580%	eta=0.110	time=225.440s
101	error=1.500%	eta=0.109	time=227.733s
102	error=1.600%	eta=0.108	time=230.024s
103	error=1.500%	eta=0.107	time=232.315s
104	error=1.570%	eta=0.105	time=234.612s
105	error=1.540%	eta=0.104	time=236.915s
106	error=1.530%	eta=0.103	time=239.194s
107	error=1.480%	eta=0.102	time=241.458s
108	error=1.520%	eta=0.101	time=243.811s
109	error=1.510%	eta=0.100	time=246.173s
110	error=1.570%	eta=0.099	time=248.518s
111	error=1.510%	eta=0.098	time=250.820s
112	error=1.500%	eta=0.097	time=253.181s
113	error=1.560%	eta=0.096	time=255.486s
114	error=1.420%	eta=0.095	time=257.771s
115	error=1.470%	eta=0.094	time=260.078s
116	error=1.490%	eta=0.093	time=262.388s
117	error=1.420%	eta=0.093	time=264.704s
118	error=1.460%	eta=0.092	time=267.010s
119	error=1.460%	eta=0.091	time=269.348s
120	error=1.480%	eta=0.090	time=271.653s
121	error=1.480%	eta=0.089	time=273.946s
122	error=1.540%	eta=0.088	time=276.404s
123	error=1.390%	eta=0.087	time=278.694s
124	error=1.460%	eta=0.086	time=280.985s
125	error=1.520%	eta=0.085	time=283.253s
126	error=1.450%	eta=0.085	time=285.586s
127	error=1.480%	eta=0.084	time=287.874s
128	error=1.460%	eta=0.083	time=290.216s
129	error=1.410%	eta=0.082	time=292.526s
130	error=1.440%	eta=0.081	time=294.774s
131	error=1.420%	eta=0.080	time=297.024s
132	error=1.430%	eta=0.080	time=299.280s
133	error=1.430%	eta=0.079	time=301.538s
134	error=1.610%	eta=0.078	time=303.799s
135	error=1.410%	eta=0.077	time=306.065s
136	error=1.440%	eta=0.076	time=308.345s
137	error=1.400%	eta=0.076	time=310.593s
138	error=1.490%	eta=0.075	time=312.829s
139	error=1.420%	eta=0.074	time=315.094s
140	error=1.380%	eta=0.073	time=317.456s
141	error=1.440%	eta=0.073	time=319.731s
142	error=1.450%	eta=0.072	time=321.988s
143	error=1.330%	eta=0.071	time=324.248s
144	error=1.440%	eta=0.071	time=326.501s
145	error=1.540%	eta=0.070	time=328.773s
146	error=1.470%	eta=0.069	time=331.070s
147	error=1.430%	eta=0.068	time=333.427s
148	error=1.490%	eta=0.068	time=335.711s
149	error=1.490%	eta=0.067	time=337.968s
150	error=1.550%	eta=0.066	time=340.217s
151	error=1.500%	eta=0.066	time=342.481s
152	error=1.470%	eta=0.065	time=344.735s
153	error=1.430%	eta=0.064	time=347.012s
154	error=1.550%	eta=0.064	time=349.257s
155	error=1.480%	eta=0.063	time=351.544s
156	error=1.440%	eta=0.063	time=353.823s
157	error=1.480%	eta=0.062	time=356.192s
158	error=1.420%	eta=0.061	time=358.460s
159	error=1.390%	eta=0.061	time=360.765s
160	error=1.420%	eta=0.060	time=363.067s
161	error=1.480%	eta=0.059	time=365.331s
162	error=1.390%	eta=0.059	time=367.625s
163	error=1.440%	eta=0.058	time=369.930s
164	error=1.480%	eta=0.058	time=372.201s
165	error=1.460%	eta=0.057	time=374.508s
166	error=1.490%	eta=0.057	time=376.806s
167	error=1.420%	eta=0.056	time=379.139s
168	error=1.400%	eta=0.055	time=381.502s
169	error=1.440%	eta=0.055	time=383.771s
170	error=1.380%	eta=0.054	time=386.146s
171	error=1.360%	eta=0.054	time=388.441s
172	error=1.400%	eta=0.053	time=390.775s
173	error=1.460%	eta=0.053	time=393.074s
174	error=1.420%	eta=0.052	time=395.377s
175	error=1.500%	eta=0.052	time=397.629s
176	error=1.400%	eta=0.051	time=399.902s
177	error=1.380%	eta=0.051	time=402.138s
178	error=1.350%	eta=0.050	time=404.391s
179	error=1.460%	eta=0.050	time=406.639s
180	error=1.380%	eta=0.049	time=408.891s
181	error=1.420%	eta=0.049	time=411.159s
182	error=1.450%	eta=0.048	time=413.448s
183	error=1.420%	eta=0.048	time=415.759s
184	error=1.400%	eta=0.047	time=418.033s
185	error=1.430%	eta=0.047	time=420.299s
186	error=1.390%	eta=0.046	time=422.576s
187	error=1.420%	eta=0.046	time=424.867s
188	error=1.340%	eta=0.045	time=427.171s
189	error=1.460%	eta=0.045	time=429.435s
190	error=1.470%	eta=0.044	time=431.668s
191	error=1.450%	eta=0.044	time=433.900s
192	error=1.350%	eta=0.044	time=436.184s
193	error=1.420%	eta=0.043	time=438.434s
194	error=1.480%	eta=0.043	time=440.754s
195	error=1.420%	eta=0.042	time=443.087s
196	error=1.380%	eta=0.042	time=445.522s
197	error=1.390%	eta=0.041	time=447.951s
198	error=1.390%	eta=0.041	time=450.294s
199	error=1.370%	eta=0.041	time=452.573s
200	error=1.400%	eta=0.040	time=454.913s
201	error=1.390%	eta=0.040	time=457.197s
202	error=1.400%	eta=0.039	time=459.491s
203	error=1.470%	eta=0.039	time=461.749s
204	error=1.310%	eta=0.039	time=464.009s

```


