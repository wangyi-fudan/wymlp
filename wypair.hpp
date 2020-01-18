#include	"wyhash.h"
#include	<math.h>
template<unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	task>
double	wypair(float	*weight,	unsigned	x,	unsigned	y,	float	t,	float	eta,	unsigned	nodes) {
	if(weight==NULL)	return	(input*2+1)*hidden+(depth-1)*hidden*hidden+1*hidden+nodes*input;
	#define	woff(i,l)	(l<0?((input*2+1)*hidden+(depth-1)*hidden*hidden+1*hidden+i*input):(l?(input*2+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden))
	#define	wypair_act(x)	(x/(1+(((int)(x>0)<<1)-1)*x))
	#define	wypair_gra(x)	((1-(((int)(x>0)<<1)-1)*x)*(1-(((int)(x>0)<<1)-1)*x))
	float	a[2*depth*hidden+1]= {},	*d=a+depth*hidden,	*o=a+2*depth*hidden,	wh=1/sqrtf(hidden),	wi=1/sqrtf(input*2+1),	*w,	s,	ds;
	for(unsigned	i=0;	i<=input*2;	i++){
		w=weight+woff(i,0);	s=i<input?weight[woff(x,-1)+i]:(i<input*2?weight[woff(y,-1)+i-input]:1);
		for(unsigned	j=0;	j<hidden;	j++)	a[j]+=s*w[j];
	}
	for(unsigned	i=0;	i<hidden;	i++) a[i]=i?wypair_act(wi*a[i]):1;
	for(unsigned	l=1;	l<=depth;	l++) {
		float	*p=a+(l-1)*hidden,	*q=(l==depth?o:a+l*hidden);
		for(unsigned	i=0;	i<(l==depth?1:hidden);	i++) {
			w=weight+woff(i,l);	s=0;
			for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
			q[i]=(l==depth?s*wh:(i?wypair_act(s*wh):1));
		}
	}
	switch(task){
	case	0:{	for(unsigned	i=0;	i<1;	i++)	if(eta<0)	return	o[i];	else	o[i]=(o[i]-t)*eta;	}	break;
	case	1:{	for(unsigned	i=0;	i<1;	i++)	if(eta<0)	return	1/(1+expf(-o[i]));	else	o[i]=(1/(1+expf(-o[i]))-t)*eta;	}	break;
	};
	for(unsigned	l=depth;	l;	l--) {
		float	*p=a+(l-1)*hidden,	*q=(l==depth?o:a+l*hidden),	*g=d+(l-1)*hidden,	*h=(l==depth?o:d+l*hidden);
		for(unsigned	i=0;	i<(l==depth?1:hidden);	i++) {
			w=weight+woff(i,l);	s=(l==depth?q[i]:h[i]*wypair_gra(q[i]))*wh;
			for(unsigned  j=0;  j<hidden; j++) {	g[j]+=s*w[j];	w[j]-=s*p[j];	}
		}
	}
	for(unsigned	i=0;	i<hidden;	i++) d[i]*=wypair_gra(a[i]);
	for(unsigned	i=0;	i<=input*2;	i++){
		w=weight+woff(i,0);	s=i<input?weight[woff(x,-1)+i]:(i<input*2?weight[woff(y,-1)+i-input]:1);	ds=0;
		for(unsigned	j=0;	j<hidden;	j++){	ds+=d[j]*w[j];	w[j]-=s*d[j];	}
		if(i<input)	weight[woff(x,-1)+i]-=ds;	else	if(i<input*2)	weight[woff(y,-1)+i-input]-=ds;
	}
	return	0;
}
/*
Author: Wang Yi <godspeed_china@yeah.net>

Example:
int	main(void){
	float	x[4]={1,2,3,5},	y[1]={2};
	vector<float>	weight(wypair<float,12,32,4,1,0>(NULL,NULL,NULL,0,0,-1));	//set dropout<0 to return size
	for(size_t	i=0;	i<weight.size();	i++)	weight[i]=3.0*rand()/RAND_MAX-1.5;
	for(unsigned	i=0;	i<1000000;	i++){
		x[0]+=0.01;	y[0]+=0.1;	//some "new" data
		wypair<float,12,32,4,1,0>(weight.data(),	x, y, 0.1,	wygrand(),	0.5);	//	training. set eta>0 to train
		wypair<float,12,32,4,1,0>(weight.data(),	x, y, -1,	wygrand(),	0.5);	//	training. set eta<0 to predict
	}
	return	0;
}

Comments:
0: task=0: regression; task=1: logistic; task=2: softmax
1: dropout<0 lead to size() function
2: eta<0 lead to prediction only.
3: The expected |X[i]|, |Y[i]| should be around 1. Normalize yor input and 1 first.
4: In practice, it is OK to call model function parallelly with multi-threads, however, they may be slower for small net.
5: The code is portable, however, if Ofast is used on X86, SSE or AVX or even AVX512 will enable very fast code!
6: The default and suggested model is shared hidden-hidden weights. If you want vanilla MLP, use the following code
if(weight==NULL)	return	(input+1)*hidden+(depth-1)*hidden*hidden+1*hidden;
#define	woff(i,l)	(l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden)

*/
