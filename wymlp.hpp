#include	"wyhash.h"
#include	<math.h>
template<class	type,	unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	output,	unsigned	loss>
struct	wymlp {
	#define	woff(i,l)	(l?(l<depth?(input+1)*hidden+i*hidden:(input+1)*hidden+hidden*hidden+i*hidden ):i*hidden)
	type	weight[(input+1)*hidden+hidden*hidden+output*hidden];
	type	act(type	x){	return	x/(1+(((int)(x>0)<<1)-1)*x);	}
	type	gra(type	x){	type	y=1-(((int)(x>0)<<1)-1)*x;	return	y*y;	}
	void	model(type	*x,	type	*y,	type	eta,	double	dropout,	uint64_t	seed) {
		type	a[2*depth*hidden+output]= {},	*d=a+depth*hidden,	*o=a+2*depth*hidden,	wh=1/sqrtf(hidden),	wi=(1-(eta<0)*dropout)/sqrtf(input+1);	uint64_t	drop=dropout*~0ull;
		for(unsigned  i=0;  i<=input; i++) if(eta<0||wyhash64(i,seed)>=drop){
			type	*w=weight+woff(i,0),	s=i==input?1:x[i];	if(s==0)	continue;
			for(unsigned	j=0;	j<hidden;	j++)	a[j]+=s*w[j];
		}
		for(unsigned	i=0;	i<hidden;	i++) a[i]=i?act(wi*a[i]):1;
		for(unsigned	l=1;	l<=depth;	l++) {
			type	*p=a+(l-1)*hidden,	*q=(l==depth?o:a+l*hidden);
			for(unsigned	i=0;	i<(l==depth?output:hidden);	i++) {
				type	*w=weight+woff(i,l),	s=0;
				for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
				q[i]=(l==depth?s*wh:(i?act(s*wh):1));
			}
		}
		switch(loss) {
		case	0:	{	for(unsigned	i=0;	i<output;	i++)	if(eta<0)	y[i]=o[i];	else	o[i]=(o[i]-y[i])*eta;	}	break;
		case	1:	{	for(unsigned	i=0;	i<output;	i++)	if(eta<0)	y[i]=1/(1+expf(-o[i]));	else	o[i]=(1/(1+expf(-o[i]))-y[i])*eta;	}	break;
		case	2:	{	type	s=0;	for(unsigned	i=0;	i<output;	i++)	s+=(o[i]=i?expf(o[i]):1);	
						for(unsigned	i=0;	i<output;	i++)	if(eta<0)	y[i]=o[i]/s;	else	o[i]=(i?(o[i]/s-(i==(unsigned)y[0])):0)*eta;	}	break;
		}
		if(eta<0) return;
		for(unsigned	l=depth;	l;	l--) {
			type	*p=a+(l-1)*hidden,	*q=(l==depth?o:a+l*hidden),	*g=d+(l-1)*hidden,	*h=(l==depth?o:d+l*hidden);
			for(unsigned	i=0;	i<(l==depth?output:hidden);	i++) {
				type	*w=weight+woff(i,l),	s=(l==depth?q[i]:h[i]*gra(q[i]))*wh;
				for(unsigned  j=0;  j<hidden; j++) {	g[j]+=s*w[j];	w[j]-=s*p[j];	}
			}
		}
		for(unsigned	i=0;	i<hidden;	i++) d[i]*=gra(a[i])*wi;
		for(unsigned  i=0;  i<=input; i++)	if(eta<0||wyhash64(i,seed)>=drop){
			type	*w=weight+woff(i,0),	s=(i==input?1:x[i]);	if(s==0)	continue;
			for(unsigned	j=0;	j<hidden;	j++)	w[j]-=s*d[j];
		}
	}
};
/*
Author: Wang Yi <godspeed_china@yeah.net>

Example:
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

Comments:
0: task=0: regression; task=1: logistic; task=2: softmax
1: eta<0 lead to prediction only.
2: The expected |X[i]|, |Y[i]| should be around 1. Normalize yor input and output first.
3: In practice, it is OK to call model function parallelly with multi-threads, however, they may be slower for small net.
4: The code is portable, however, if Ofast is used on X86, SSE or AVX or even AVX512 will enable very fast code!
5: The default and suggested model is shared hidden-hidden weights. If you want conventional MLP, please replace it with the following lines:
	#define	woff(i,l)	(l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden)
	type	weight[(input+1)*hidden+(depth-1)*hidden*hidden+output*hidden];
*/
