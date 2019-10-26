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

## MNIST 128H8L with single CPU thread:

```
0	error=8.120%	eta=0.300	time=4.743s
1	error=5.900%	eta=0.297	time=9.508s
2	error=5.020%	eta=0.294	time=14.257s
3	error=4.340%	eta=0.291	time=19.012s
4	error=4.800%	eta=0.288	time=23.761s
5	error=3.530%	eta=0.285	time=28.521s
6	error=3.420%	eta=0.282	time=33.276s
7	error=3.820%	eta=0.280	time=38.028s
8	error=3.260%	eta=0.277	time=42.795s
9	error=3.060%	eta=0.274	time=47.561s
10	error=2.930%	eta=0.271	time=52.348s
11	error=2.970%	eta=0.269	time=57.119s
12	error=2.710%	eta=0.266	time=61.880s
13	error=2.840%	eta=0.263	time=66.634s
14	error=2.570%	eta=0.261	time=71.397s
15	error=2.670%	eta=0.258	time=76.160s
16	error=2.640%	eta=0.255	time=80.925s
17	error=2.370%	eta=0.253	time=85.679s
18	error=2.310%	eta=0.250	time=90.443s
19	error=2.370%	eta=0.248	time=95.210s
20	error=2.210%	eta=0.245	time=100.001s
21	error=2.540%	eta=0.243	time=104.792s
22	error=2.220%	eta=0.240	time=109.603s
23	error=2.280%	eta=0.238	time=114.402s
24	error=2.180%	eta=0.236	time=119.230s
25	error=2.280%	eta=0.233	time=124.026s
26	error=2.920%	eta=0.231	time=128.878s
27	error=1.990%	eta=0.229	time=133.699s
28	error=1.980%	eta=0.226	time=138.507s
29	error=2.050%	eta=0.224	time=143.320s
30	error=2.230%	eta=0.222	time=148.125s
31	error=2.000%	eta=0.220	time=152.949s
32	error=2.090%	eta=0.217	time=157.754s
33	error=1.940%	eta=0.215	time=162.561s
34	error=1.960%	eta=0.213	time=167.436s
35	error=1.860%	eta=0.211	time=172.257s
36	error=1.920%	eta=0.209	time=177.055s
37	error=2.000%	eta=0.207	time=181.857s
38	error=1.840%	eta=0.205	time=186.689s
39	error=1.800%	eta=0.203	time=191.505s
40	error=1.740%	eta=0.201	time=196.340s
41	error=1.980%	eta=0.199	time=201.153s
42	error=1.810%	eta=0.197	time=205.957s
43	error=1.910%	eta=0.195	time=210.781s
44	error=1.800%	eta=0.193	time=215.564s
45	error=2.000%	eta=0.191	time=220.397s
46	error=1.910%	eta=0.189	time=225.194s
47	error=1.850%	eta=0.187	time=230.012s
48	error=1.890%	eta=0.185	time=234.808s
49	error=1.840%	eta=0.183	time=239.629s
50	error=1.800%	eta=0.182	time=244.435s
51	error=1.730%	eta=0.180	time=249.258s
52	error=1.780%	eta=0.178	time=254.083s
53	error=1.790%	eta=0.176	time=258.910s
54	error=1.910%	eta=0.174	time=263.725s
55	error=1.880%	eta=0.173	time=268.501s
56	error=1.880%	eta=0.171	time=273.298s
57	error=1.710%	eta=0.169	time=278.122s
58	error=1.730%	eta=0.167	time=282.951s
59	error=1.570%	eta=0.166	time=287.767s
60	error=1.920%	eta=0.164	time=292.581s
61	error=1.730%	eta=0.163	time=297.387s
62	error=1.560%	eta=0.161	time=302.192s
63	error=1.820%	eta=0.159	time=306.982s
64	error=1.750%	eta=0.158	time=311.792s
65	error=2.080%	eta=0.156	time=316.592s
66	error=1.550%	eta=0.155	time=321.395s
67	error=1.770%	eta=0.153	time=326.194s
68	error=1.890%	eta=0.151	time=331.002s
69	error=1.710%	eta=0.150	time=335.826s
70	error=1.830%	eta=0.148	time=340.642s
71	error=1.730%	eta=0.147	time=345.434s
72	error=1.520%	eta=0.145	time=350.268s
73	error=1.690%	eta=0.144	time=355.091s
74	error=1.690%	eta=0.143	time=359.898s
75	error=1.740%	eta=0.141	time=364.710s
76	error=1.910%	eta=0.140	time=369.513s
77	error=1.770%	eta=0.138	time=374.336s
78	error=1.760%	eta=0.137	time=379.146s
79	error=1.820%	eta=0.136	time=383.964s
80	error=1.670%	eta=0.134	time=388.772s
81	error=1.810%	eta=0.133	time=393.575s
82	error=1.610%	eta=0.132	time=398.385s
83	error=1.730%	eta=0.130	time=403.191s
84	error=1.600%	eta=0.129	time=408.006s
85	error=1.610%	eta=0.128	time=412.799s
86	error=1.660%	eta=0.126	time=417.593s
87	error=1.720%	eta=0.125	time=422.404s
88	error=1.700%	eta=0.124	time=427.196s
89	error=1.590%	eta=0.123	time=432.026s
90	error=1.690%	eta=0.121	time=436.830s
91	error=1.530%	eta=0.120	time=441.637s
92	error=1.730%	eta=0.119	time=446.482s
93	error=1.750%	eta=0.118	time=451.286s
94	error=1.810%	eta=0.117	time=456.081s
95	error=1.660%	eta=0.115	time=460.906s
96	error=1.620%	eta=0.114	time=465.730s
97	error=1.530%	eta=0.113	time=470.562s
98	error=1.620%	eta=0.112	time=475.372s
99	error=1.560%	eta=0.111	time=480.202s
100	error=1.730%	eta=0.110	time=484.991s
101	error=1.610%	eta=0.109	time=489.791s
102	error=1.640%	eta=0.108	time=494.608s
103	error=1.540%	eta=0.107	time=499.413s
104	error=1.740%	eta=0.105	time=504.224s
105	error=1.510%	eta=0.104	time=509.048s
106	error=1.750%	eta=0.103	time=513.851s
107	error=1.630%	eta=0.102	time=518.649s
108	error=1.490%	eta=0.101	time=523.491s
109	error=1.600%	eta=0.100	time=528.315s
110	error=1.870%	eta=0.099	time=533.123s
111	error=1.560%	eta=0.098	time=537.945s
112	error=1.560%	eta=0.097	time=542.759s
113	error=1.590%	eta=0.096	time=547.590s
114	error=1.820%	eta=0.095	time=552.409s
115	error=1.620%	eta=0.094	time=557.213s
116	error=1.680%	eta=0.093	time=562.027s
117	error=1.490%	eta=0.093	time=566.878s
118	error=1.590%	eta=0.092	time=571.699s
119	error=1.600%	eta=0.091	time=576.523s
120	error=1.740%	eta=0.090	time=581.330s
121	error=1.780%	eta=0.089	time=586.160s
122	error=1.740%	eta=0.088	time=590.979s
123	error=1.660%	eta=0.087	time=595.834s
124	error=1.610%	eta=0.086	time=600.689s
125	error=1.670%	eta=0.085	time=605.516s
126	error=1.640%	eta=0.085	time=610.336s
127	error=1.620%	eta=0.084	time=615.139s
128	error=1.630%	eta=0.083	time=619.910s
129	error=1.680%	eta=0.082	time=624.722s
130	error=1.690%	eta=0.081	time=629.496s
131	error=1.690%	eta=0.080	time=634.266s
132	error=1.620%	eta=0.080	time=639.132s
133	error=1.610%	eta=0.079	time=643.982s
134	error=1.760%	eta=0.078	time=648.850s
135	error=1.650%	eta=0.077	time=653.732s
136	error=1.620%	eta=0.076	time=658.549s
137	error=1.530%	eta=0.076	time=663.376s
138	error=1.570%	eta=0.075	time=668.238s
139	error=1.520%	eta=0.074	time=673.067s
140	error=1.580%	eta=0.073	time=677.918s
141	error=1.560%	eta=0.073	time=682.761s
142	error=1.530%	eta=0.072	time=687.602s
143	error=1.510%	eta=0.071	time=692.468s
144	error=1.620%	eta=0.071	time=697.325s
145	error=1.580%	eta=0.070	time=702.185s
146	error=1.650%	eta=0.069	time=706.894s
147	error=1.580%	eta=0.068	time=711.734s
148	error=1.660%	eta=0.068	time=716.583s
149	error=1.640%	eta=0.067	time=721.447s
150	error=1.680%	eta=0.066	time=726.313s
151	error=1.630%	eta=0.066	time=731.161s
152	error=1.760%	eta=0.065	time=736.006s
153	error=1.730%	eta=0.064	time=740.875s
154	error=1.620%	eta=0.064	time=745.710s
155	error=1.740%	eta=0.063	time=750.567s
156	error=1.590%	eta=0.063	time=755.407s
157	error=1.720%	eta=0.062	time=760.274s
158	error=1.640%	eta=0.061	time=765.107s
159	error=1.610%	eta=0.061	time=769.977s
160	error=1.780%	eta=0.060	time=774.826s
161	error=1.650%	eta=0.059	time=779.692s
162	error=1.500%	eta=0.059	time=784.576s
163	error=1.660%	eta=0.058	time=789.425s
164	error=1.540%	eta=0.058	time=794.278s
165	error=1.590%	eta=0.057	time=799.128s
166	error=1.670%	eta=0.057	time=803.975s
167	error=1.630%	eta=0.056	time=808.799s
168	error=1.630%	eta=0.055	time=813.660s
169	error=1.590%	eta=0.055	time=818.495s
170	error=1.680%	eta=0.054	time=823.368s
171	error=1.630%	eta=0.054	time=828.252s
172	error=1.540%	eta=0.053	time=833.097s
173	error=1.570%	eta=0.053	time=837.960s
174	error=1.660%	eta=0.052	time=842.831s
175	error=1.630%	eta=0.052	time=847.696s
176	error=1.650%	eta=0.051	time=852.566s
177	error=1.520%	eta=0.051	time=857.417s
178	error=1.640%	eta=0.050	time=862.286s
179	error=1.620%	eta=0.050	time=867.131s
180	error=1.620%	eta=0.049	time=871.986s
181	error=1.590%	eta=0.049	time=876.829s
182	error=1.590%	eta=0.048	time=881.684s
183	error=1.620%	eta=0.048	time=886.530s
184	error=1.710%	eta=0.047	time=891.410s
185	error=1.580%	eta=0.047	time=896.270s
186	error=1.740%	eta=0.046	time=901.128s
187	error=1.650%	eta=0.046	time=906.005s
188	error=1.610%	eta=0.045	time=910.860s
189	error=1.580%	eta=0.045	time=915.680s
190	error=1.630%	eta=0.044	time=920.522s
191	error=1.620%	eta=0.044	time=925.359s
192	error=1.630%	eta=0.044	time=930.190s
193	error=1.630%	eta=0.043	time=935.027s
194	error=1.640%	eta=0.043	time=939.880s
195	error=1.580%	eta=0.042	time=944.719s
196	error=1.590%	eta=0.042	time=949.569s
197	error=1.680%	eta=0.041	time=954.418s
198	error=1.570%	eta=0.041	time=959.286s
199	error=1.700%	eta=0.041	time=964.177s
200	error=1.650%	eta=0.040	time=969.012s
201	error=1.610%	eta=0.040	time=973.871s
202	error=1.710%	eta=0.039	time=978.701s
203	error=1.620%	eta=0.039	time=983.556s
204	error=1.670%	eta=0.039	time=988.390s
205	error=1.630%	eta=0.038	time=993.238s
206	error=1.510%	eta=0.038	time=998.091s
207	error=1.620%	eta=0.037	time=1002.946s
208	error=1.620%	eta=0.037	time=1007.786s
209	error=1.640%	eta=0.037	time=1012.635s
210	error=1.650%	eta=0.036	time=1017.490s
211	error=1.560%	eta=0.036	time=1022.320s
212	error=1.630%	eta=0.036	time=1027.213s
213	error=1.630%	eta=0.035	time=1032.096s
214	error=1.720%	eta=0.035	time=1037.016s
215	error=1.660%	eta=0.035	time=1041.873s
216	error=1.640%	eta=0.034	time=1046.727s
217	error=1.610%	eta=0.034	time=1051.580s
218	error=1.610%	eta=0.034	time=1056.431s
219	error=1.660%	eta=0.033	time=1061.283s
220	error=1.700%	eta=0.033	time=1066.137s
221	error=1.650%	eta=0.033	time=1070.999s
222	error=1.630%	eta=0.032	time=1075.842s
223	error=1.540%	eta=0.032	time=1080.682s
224	error=1.590%	eta=0.032	time=1085.529s
225	error=1.580%	eta=0.031	time=1090.403s
226	error=1.590%	eta=0.031	time=1095.254s
227	error=1.550%	eta=0.031	time=1100.103s
228	error=1.690%	eta=0.030	time=1104.956s
229	error=1.740%	eta=0.030	time=1109.805s
230	error=1.660%	eta=0.030	time=1114.665s
231	error=1.590%	eta=0.029	time=1119.526s
232	error=1.620%	eta=0.029	time=1124.366s
233	error=1.660%	eta=0.029	time=1129.220s
234	error=1.530%	eta=0.029	time=1134.066s
235	error=1.600%	eta=0.028	time=1138.929s
236	error=1.620%	eta=0.028	time=1143.802s
237	error=1.630%	eta=0.028	time=1148.641s
238	error=1.590%	eta=0.027	time=1153.484s
239	error=1.640%	eta=0.027	time=1158.341s
240	error=1.500%	eta=0.027	time=1163.175s
241	error=1.600%	eta=0.027	time=1168.031s
242	error=1.570%	eta=0.026	time=1172.903s
243	error=1.650%	eta=0.026	time=1177.768s
244	error=1.560%	eta=0.026	time=1182.611s
245	error=1.630%	eta=0.026	time=1187.454s
246	error=1.680%	eta=0.025	time=1192.287s
247	error=1.680%	eta=0.025	time=1197.170s
248	error=1.540%	eta=0.025	time=1202.020s
249	error=1.540%	eta=0.025	time=1206.846s
250	error=1.580%	eta=0.024	time=1211.660s
251	error=1.600%	eta=0.024	time=1216.513s
252	error=1.610%	eta=0.024	time=1221.355s
253	error=1.540%	eta=0.024	time=1226.202s
254	error=1.580%	eta=0.023	time=1231.062s
255	error=1.530%	eta=0.023	time=1235.942s
256	error=1.590%	eta=0.023	time=1240.791s
257	error=1.570%	eta=0.023	time=1245.654s
258	error=1.630%	eta=0.022	time=1250.528s
259	error=1.580%	eta=0.022	time=1255.384s
260	error=1.650%	eta=0.022	time=1260.230s
261	error=1.640%	eta=0.022	time=1265.081s
262	error=1.590%	eta=0.022	time=1269.938s
263	error=1.590%	eta=0.021	time=1274.794s
264	error=1.610%	eta=0.021	time=1279.643s
265	error=1.560%	eta=0.021	time=1284.485s
266	error=1.610%	eta=0.021	time=1289.318s
267	error=1.600%	eta=0.020	time=1294.191s
268	error=1.610%	eta=0.020	time=1299.069s
269	error=1.580%	eta=0.020	time=1303.907s
270	error=1.640%	eta=0.020	time=1308.777s
271	error=1.630%	eta=0.020	time=1313.617s
272	error=1.570%	eta=0.019	time=1318.478s
273	error=1.590%	eta=0.019	time=1323.326s
274	error=1.620%	eta=0.019	time=1328.184s
275	error=1.580%	eta=0.019	time=1333.043s
276	error=1.580%	eta=0.019	time=1337.907s
277	error=1.650%	eta=0.019	time=1342.729s
278	error=1.680%	eta=0.018	time=1347.557s
279	error=1.630%	eta=0.018	time=1352.402s
280	error=1.600%	eta=0.018	time=1357.244s
281	error=1.580%	eta=0.018	time=1362.074s
282	error=1.640%	eta=0.018	time=1366.920s
283	error=1.620%	eta=0.017	time=1371.802s
284	error=1.600%	eta=0.017	time=1376.667s
285	error=1.640%	eta=0.017	time=1381.518s
286	error=1.540%	eta=0.017	time=1386.343s
287	error=1.580%	eta=0.017	time=1391.185s
288	error=1.600%	eta=0.017	time=1396.034s
289	error=1.670%	eta=0.016	time=1400.870s
290	error=1.570%	eta=0.016	time=1405.699s
291	error=1.620%	eta=0.016	time=1410.566s
292	error=1.630%	eta=0.016	time=1415.395s
293	error=1.620%	eta=0.016	time=1420.270s
294	error=1.590%	eta=0.016	time=1425.140s
295	error=1.560%	eta=0.015	time=1430.008s
296	error=1.620%	eta=0.015	time=1434.911s
297	error=1.650%	eta=0.015	time=1439.784s
298	error=1.570%	eta=0.015	time=1444.681s
299	error=1.570%	eta=0.015	time=1449.531s
300	error=1.660%	eta=0.015	time=1454.392s
301	error=1.530%	eta=0.015	time=1459.230s
302	error=1.610%	eta=0.014	time=1464.083s
303	error=1.560%	eta=0.014	time=1468.943s
304	error=1.520%	eta=0.014	time=1473.798s
305	error=1.640%	eta=0.014	time=1478.636s
306	error=1.570%	eta=0.014	time=1483.474s
307	error=1.580%	eta=0.014	time=1488.328s
308	error=1.600%	eta=0.014	time=1493.196s
309	error=1.560%	eta=0.013	time=1498.039s
310	error=1.590%	eta=0.013	time=1502.878s
311	error=1.600%	eta=0.013	time=1507.717s
312	error=1.580%	eta=0.013	time=1512.556s
313	error=1.580%	eta=0.013	time=1517.401s
314	error=1.610%	eta=0.013	time=1522.229s
315	error=1.590%	eta=0.013	time=1527.068s
316	error=1.550%	eta=0.013	time=1531.921s
317	error=1.570%	eta=0.012	time=1536.743s
318	error=1.600%	eta=0.012	time=1541.622s
319	error=1.590%	eta=0.012	time=1546.473s
320	error=1.560%	eta=0.012	time=1551.317s
321	error=1.540%	eta=0.012	time=1556.162s
322	error=1.560%	eta=0.012	time=1561.002s
323	error=1.540%	eta=0.012	time=1565.867s
324	error=1.540%	eta=0.012	time=1570.702s
325	error=1.490%	eta=0.011	time=1575.539s
326	error=1.530%	eta=0.011	time=1580.379s
327	error=1.550%	eta=0.011	time=1585.223s
328	error=1.570%	eta=0.011	time=1590.084s
329	error=1.580%	eta=0.011	time=1594.766s
330	error=1.610%	eta=0.011	time=1599.617s
331	error=1.600%	eta=0.011	time=1604.470s
332	error=1.540%	eta=0.011	time=1609.315s
333	error=1.570%	eta=0.011	time=1614.136s
334	error=1.660%	eta=0.010	time=1618.954s
335	error=1.560%	eta=0.010	time=1623.831s
336	error=1.520%	eta=0.010	time=1628.678s
337	error=1.610%	eta=0.010	time=1633.513s
338	error=1.600%	eta=0.010	time=1638.373s
339	error=1.600%	eta=0.010	time=1643.248s
340	error=1.560%	eta=0.010	time=1648.153s
341	error=1.660%	eta=0.010	time=1653.015s
342	error=1.660%	eta=0.010	time=1657.891s
343	error=1.590%	eta=0.010	time=1662.844s
344	error=1.520%	eta=0.009	time=1667.697s
345	error=1.570%	eta=0.009	time=1672.547s
346	error=1.540%	eta=0.009	time=1677.432s
347	error=1.590%	eta=0.009	time=1682.279s
348	error=1.650%	eta=0.009	time=1687.123s
349	error=1.550%	eta=0.009	time=1691.952s
350	error=1.570%	eta=0.009	time=1696.801s
351	error=1.580%	eta=0.009	time=1701.624s
352	error=1.600%	eta=0.009	time=1706.456s
353	error=1.520%	eta=0.009	time=1711.330s
354	error=1.560%	eta=0.009	time=1716.173s
355	error=1.550%	eta=0.008	time=1721.009s
356	error=1.560%	eta=0.008	time=1725.850s
357	error=1.500%	eta=0.008	time=1730.723s
358	error=1.480%	eta=0.008	time=1735.576s
359	error=1.570%	eta=0.008	time=1740.412s
360	error=1.520%	eta=0.008	time=1745.259s
361	error=1.490%	eta=0.008	time=1750.117s
362	error=1.520%	eta=0.008	time=1754.942s
363	error=1.600%	eta=0.008	time=1759.801s
364	error=1.540%	eta=0.008	time=1764.636s
365	error=1.540%	eta=0.008	time=1769.484s
366	error=1.530%	eta=0.008	time=1774.341s
367	error=1.530%	eta=0.008	time=1779.212s
368	error=1.570%	eta=0.007	time=1784.049s
369	error=1.550%	eta=0.007	time=1788.882s
370	error=1.540%	eta=0.007	time=1793.706s
371	error=1.620%	eta=0.007	time=1798.554s
372	error=1.570%	eta=0.007	time=1803.403s
373	error=1.500%	eta=0.007	time=1808.264s
374	error=1.550%	eta=0.007	time=1813.142s
375	error=1.510%	eta=0.007	time=1818.001s
376	error=1.550%	eta=0.007	time=1822.881s
377	error=1.550%	eta=0.007	time=1827.742s
378	error=1.560%	eta=0.007	time=1832.616s
379	error=1.590%	eta=0.007	time=1837.445s
380	error=1.560%	eta=0.007	time=1842.309s
381	error=1.570%	eta=0.007	time=1847.148s
382	error=1.580%	eta=0.006	time=1851.968s
383	error=1.580%	eta=0.006	time=1856.726s
384	error=1.540%	eta=0.006	time=1861.479s
385	error=1.600%	eta=0.006	time=1866.237s
386	error=1.530%	eta=0.006	time=1870.997s
387	error=1.570%	eta=0.006	time=1875.759s
388	error=1.560%	eta=0.006	time=1880.517s
389	error=1.530%	eta=0.006	time=1885.282s
390	error=1.560%	eta=0.006	time=1890.042s
391	error=1.560%	eta=0.006	time=1894.819s
392	error=1.600%	eta=0.006	time=1899.576s
393	error=1.530%	eta=0.006	time=1904.325s
394	error=1.590%	eta=0.006	time=1909.100s
395	error=1.580%	eta=0.006	time=1913.854s
396	error=1.580%	eta=0.006	time=1918.617s
397	error=1.560%	eta=0.006	time=1923.389s
398	error=1.530%	eta=0.005	time=1928.156s
399	error=1.590%	eta=0.005	time=1932.918s
400	error=1.550%	eta=0.005	time=1937.675s
401	error=1.530%	eta=0.005	time=1942.430s
402	error=1.560%	eta=0.005	time=1947.207s
403	error=1.600%	eta=0.005	time=1951.988s
404	error=1.620%	eta=0.005	time=1956.743s
405	error=1.630%	eta=0.005	time=1961.516s
406	error=1.570%	eta=0.005	time=1966.274s
407	error=1.600%	eta=0.005	time=1971.034s
408	error=1.550%	eta=0.005	time=1975.802s
409	error=1.580%	eta=0.005	time=1980.580s
410	error=1.590%	eta=0.005	time=1985.334s
411	error=1.550%	eta=0.005	time=1990.095s
412	error=1.580%	eta=0.005	time=1994.873s
413	error=1.550%	eta=0.005	time=1999.655s
414	error=1.600%	eta=0.005	time=2004.424s
415	error=1.560%	eta=0.005	time=2009.169s
416	error=1.570%	eta=0.005	time=2013.936s
417	error=1.570%	eta=0.005	time=2018.689s
418	error=1.560%	eta=0.004	time=2023.447s
419	error=1.530%	eta=0.004	time=2028.213s
420	error=1.530%	eta=0.004	time=2032.971s
421	error=1.550%	eta=0.004	time=2037.747s
```


