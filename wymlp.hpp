//Author: Wang Yi <godspeed_china@yeah.net>
#include	"wyhash.h"
#include	<cstdio>
#include	<cmath>
using	namespace	std;
template<class	type,	unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	output,	unsigned	loss>
class	wymlp {
private:
#ifdef	SharedHiddenMatrix
	#define	wymlp_size	((input+1)*hidden+hidden*hidden+output*hidden)
	unsigned	woff(unsigned	i,	unsigned	l) {	return	l?(l<depth?(input+1)*hidden+i*hidden:(input+1)*hidden+hidden*hidden+i*hidden ):i*hidden;	}
#else
	#define	wymlp_size	((input+1)*hidden+(depth-1)*hidden*hidden+output*hidden)
	unsigned	woff(unsigned	i,	unsigned	l) {	return	l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden;	}
#endif
	type	weight[wymlp_size],	work[2*depth*hidden+output];
	type	act(type	x) {	return	x>0?x/(1+x):x/(1-x);	}
	type	gra(type	x) {	return	x>0?(1-x)*(1-x):(1+x)*(1+x);	}	
public:
	void	random(uint64_t	s) {	for(unsigned	i=0;	i<wymlp_size;	i++)	weight[i]=wy2gau(wyrand(&s));	}
	bool	save(const	char	*F) {
		FILE	*f=fopen(F,	"wb");	if(f==NULL)	return	0;
		if(fwrite(weight,	wymlp_size*sizeof(type),	1,	f)!=1)	return	0;
		fclose(f);	return	1;
	}
	bool	load(const	char	*F) {
		FILE	*f=fopen(F,	"rb");	if(f==NULL)	return	0;
		if(fread(weight,	wymlp_size*sizeof(type),	1,	f)!=1)	return	0;
		fclose(f);	return	1;
	}
	void	model(type	*x,	type	*y,	type	eta) {
		type	*p,	*q,	*o,	*w,	*g,	*h,	*a=work,	*d=a+depth*hidden,	s,	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
		memset(work,	0,	(2*depth*hidden+output)*sizeof(type));
		p=work;
		for(unsigned  i=0;  i<=input; i++) {
			w=weight+woff(i,0);	s=i==input?1:x[i];	if(s==0)	continue;
			for(unsigned	j=0;	j<hidden;	j++)	p[j]+=s*w[j];
		}
		for(unsigned	i=0;	i<hidden;	i++)	p[i]=i?act(wi*p[i]):1;
		for(unsigned	l=1;	l<depth;	l++) {
			p=a+(l-1)*hidden;	q=a+l*hidden;
			for(unsigned	i=0;	i<hidden;	i++) {
				s=0;	w=weight+woff(i,l);
				for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
				q[i]=i?act(wh*s):1;
			}
		}
		o=a+2*depth*hidden;	p=a+(depth-1)*hidden;
		for(unsigned	i=0;	i<output;	i++) {
			s=0;	w=weight+woff(i,depth);
			for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
			o[i]=wh*s;
		}
		switch(loss) {
		case	0:	{	for(unsigned	i=0;	i<output;	i++)	if(eta<0)	y[i]=o[i];	else	o[i]-=y[i];	}	break;
		case	1:	{	for(unsigned	i=0;	i<output;	i++)	if(eta<0)	y[i]=1/(1+expf(-o[i]));	else	o[i]=1/(1+expf(-o[i]))-y[i];	}	break;
		case	2:	{	for(unsigned	i=s=0;	i<output;	i++)	s+=(o[i]=i?expf(o[i]):1);	
					for(unsigned	i=0;	i<output;	i++)	if(eta<0)	y[i]=o[i]/s;	else	o[i]=i?(o[i]/s-(i==(unsigned)y[0])):0;	}	break;
		}
		if(eta<0) return;
		for(unsigned	i=0;	i<output;	i++) {
			w=weight+woff(i,depth);	p=a+(depth-1)*hidden;	g=d+(depth-1)*hidden;	s=o[i]*wh*eta;
			for(unsigned  j=0;  j<hidden; j++) {	g[j]+=s*w[j];	w[j]-=s*p[j];	}
		}
		for(unsigned	l=depth-1;	l;	l--) {
			p=a+(l-1)*hidden;	q=a+(l)*hidden;	g=d+(l-1)*hidden;	h=d+l*hidden;
			for(unsigned	i=0;	i<hidden;	i++) {
				w=weight+woff(i,l);	s=h[i]*gra(q[i])*wh;
				for(unsigned  j=0;  j<hidden; j++) {	g[j]+=s*w[j];	w[j]-=s*p[j];	}
			}
		}
		p=work;	g=d;
		for(unsigned	i=0;	i<hidden;	i++)	g[i]*=gra(p[i])*wi;
		for(unsigned  i=0;  i<=input; i++)	{
			w=weight+woff(i,0);	s=(i==input?1:x[i]);	if(s==0)	continue;
			for(unsigned	j=0;	j<hidden;	j++)	w[j]-=s*g[j];
		}
	}
};
/*	Example:
	wymlp<float,4,16,3,1,0>	model;	//	task=0: regression; task=1: logistic;	task=2:	softmax
	model.ramdom(time(NULL));
	float	x[4]={1,2,3,5},	y[1]={2};
	model.model(x,y,0.1);	//	to learn x-y pair
	model.model(x,y,-1);	//	set eta<0 to predict x, and store to y
	model.save("model");
*/