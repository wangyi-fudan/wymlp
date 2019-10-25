//Author: Wang Yi <godspeed_china@yeah.net>	Documents are at to bottom.
#include	<stdlib.h>
#include	<stdio.h>
#include	<math.h>
template<class	type,	unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	output,	unsigned	loss>
class	wymlp {
private:
	#define	wymlp_size	((input+1)*hidden+hidden*hidden+output*hidden)
	unsigned	woff(unsigned	i,	unsigned	l) {	return	l?(l<depth?(input+1)*hidden+i*hidden:(input+1)*hidden+hidden*hidden+i*hidden ):i*hidden;	}
	type	weight[wymlp_size];
public:
	void	random(unsigned	s) {	srand(s);	for(unsigned	i=0;	i<wymlp_size;	i++)	weight[i]=3.0*rand()/RAND_MAX-1.5;	}
	void	save(const	char	*F) {	FILE	*f=fopen(F,	"wb");	if(fwrite(weight,	wymlp_size*sizeof(type),	1,	f)!=1)	return;	fclose(f);	}
	void	load(const	char	*F) {	FILE	*f=fopen(F,	"rb");	if(fread(weight,	wymlp_size*sizeof(type),	1,	f)!=1)	return;	fclose(f);	}
	void	model(type	*x,	type	*y,	type	eta) {
		type	a[2*depth*hidden+output]= {},	*d=a+depth*hidden,	*o=a+2*depth*hidden,	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
		for(unsigned  i=0;  i<=input; i++) {
			type	*w=weight+woff(i,0),	s=i==input?1:x[i];	if(s==0)	continue;
			for(unsigned	j=0;	j<hidden;	j++)	a[j]+=s*w[j];
		}
		for(unsigned	i=0;	i<hidden;	i++) {	type	s=wi*a[i];	a[i]=i?(s>0?s/(1+s):s/(1-s)):1;	}
		for(unsigned	l=1;	l<=depth;	l++) {
			type	*p=a+(l-1)*hidden,	*q=(l==depth?o:a+l*hidden);
			for(unsigned	i=0;	i<(l==depth?output:hidden);	i++) {
				type	*w=weight+woff(i,l),	s=0;
				for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
				s*=wh;	q[i]=(l==depth?s:(i?(s>0?s/(1+s):s/(1-s)):1));
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
				type	*w=weight+woff(i,l),	s=q[i];	s=(l==depth?s:h[i]*(s>0?(1-s)*(1-s):(1+s)*(1+s)))*wh;
				for(unsigned  j=0;  j<hidden; j++) {	g[j]+=s*w[j];	w[j]-=s*p[j];	}
			}
		}
		for(unsigned	i=0;	i<hidden;	i++) {	type	s=a[i];	d[i]*=(s>0?(1-s)*(1-s):(1+s)*(1+s))*wi;	}
		for(unsigned  i=0;  i<=input; i++)	{
			type	*w=weight+woff(i,0),	s=(i==input?1:x[i]);	if(s==0)	continue;
			for(unsigned	j=0;	j<hidden;	j++)	w[j]-=s*d[j];
		}
	}
};
/*
Example:
	wymlp<float,4,16,3,1,0>	model;	
	model.ramdom(time(NULL));
	float	x[4]={1,2,3,5},	y[1]={2};
	model.model(x, y, 0.1);	//	train
	model.model(x, y, -1);	//	predict
	model.save("model");
Comments:
0: task=0: regression; task=1: logistic; task=2: softmax
1: eta<0 lead to prediction only.
2: The expected |X[i]|, |Y[i]| should be around 1. Normalize yor input and output first.
3: In practice, it is OK to call model function parallelly with multi-threads, however, they may be slower for small net.
4: The code is portable, however, if O3 is used on X86, SSE or AVX or even AVX512 will enable very fast code!
5: The default and suggested model is shared hidden-hidden weights. If you want conventional MLP, please replace it with the following lines:
	#define	wymlp_size	((input+1)*hidden+(depth-1)*hidden*hidden+output*hidden)
	unsigned	woff(unsigned	i,	unsigned	l) {	return	l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden;	}
*/