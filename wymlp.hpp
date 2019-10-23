//Author: Wang Yi <godspeed_china@yeah.net>
#include	"wyhash.h"
#include	<cstdlib>
#include	<cstdio>
#include	<cmath>
using	namespace	std;
template<class	type,	unsigned	input,	unsigned	hidden,	unsigned	layer,	unsigned	output,	unsigned	loss>	//loss:		0:squared loss	1:logistic loss	2:softmax loss
class	wymlp {
private:
	type	drop,	*weight,	*work;
	type	act(type	x) {	return	x>0?x/(1+x):x/(1-x);	}
	type	gra(type	x) {	return	x>0?(1-x)*(1-x):(1+x)*(1+x);	}
	unsigned	size(void) {	return	(input+1)*hidden+(layer-1)*hidden*hidden+output*hidden;	}
	unsigned	woff(unsigned	i,	unsigned	l) {	return	l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden;	}
	unsigned	aoff(unsigned	l) {	return	l*hidden;	}
	unsigned	doff(unsigned	l) {	return	layer*hidden+l*hidden;	}
public:
	wymlp(type	dropout) {	drop=dropout;	weight=(type*)malloc(size()*sizeof(type));	work=(type*)malloc((2*layer*hidden+output)*sizeof(type));	}
	~wymlp() {	free(weight);	free(work);	}
	void	random(uint64_t	s) {	for(unsigned	i=0;	i<size();	i++)	weight[i]=wy2gau(wyrand(&s));	}
	bool	save(const	char	*F) {
		FILE	*f=fopen(F,	"wb");	if(f==NULL)	return	0;
		unsigned	n;
		n=input;	if(fwrite(&n,	sizeof(unsigned),	1,	f)!=1)	return	0;
		n=hidden;	if(fwrite(&n,	sizeof(unsigned),	1,	f)!=1)	return	0;
		n=layer;	if(fwrite(&n,	sizeof(unsigned),	1,	f)!=1)	return	0;
		n=output;	if(fwrite(&n,	sizeof(unsigned),	1,	f)!=1)	return	0;
		n=loss;		if(fwrite(&n,	sizeof(unsigned),	1,	f)!=1)	return	0;
		if(fwrite(&drop,	sizeof(type),	1,	f)!=1)	return	0;
		if(fwrite(weight,	size()*sizeof(type),	1,	f)!=1)	return	0;
		fclose(f);	return	1;
	}
	bool	load(const	char	*F) {
		FILE	*f=fopen(F,	"rb");	if(f==NULL)	return	0;
		unsigned	n;
		if(fread(&n,	sizeof(unsigned),	1,	f)!=1||n!=input)	return	0;
		if(fread(&n,	sizeof(unsigned),	1,	f)!=1||n!=hidden)	return	0;
		if(fread(&n,	sizeof(unsigned),	1,	f)!=1||n!=layer)	return	0;
		if(fread(&n,	sizeof(unsigned),	1,	f)!=1||n!=output)	return	0;
		if(fread(&n,	sizeof(unsigned),	1,	f)!=1||n!=loss)	return	0;
		if(fread(&drop,	sizeof(type),	1,	f)!=1)	return	0;
		if(fread(weight,	size()*sizeof(type),	1,	f)!=1)	return	0;
		fclose(f);	return	1;
	}
	void	model(type	*x,	type	*y,	type	alpha,	uint64_t	seed) {	//	set alpha<0 to predict. x and y are suggested to be standized
		type	*p,	*q,	*o,	*w,	*g,	*h,	s,	wh=1/sqrtf(hidden),	wi=(1-(alpha<0)*drop)/sqrtf(input+1);
		memset(work,	0,	(2*layer*hidden+output)*sizeof(type));
		p=work+aoff(0);	
		for(unsigned  i=0;  i<=input; i++) if(alpha<0||wy2u01(wyhash64(i,seed))>=drop) {
			s=i==input?1:x[i];	if(s==0)	continue;	w=weight+woff(i,0);
			for(unsigned	j=0;	j<hidden;	j++)	p[j]+=s*w[j];
		}
		for(unsigned	i=0;	i<hidden;	i++)	p[i]=i?act(wi*p[i]):1;
		for(unsigned	l=1;	l<layer;	l++) {
			p=work+aoff(l-1);	q=work+aoff(l);
			for(unsigned	i=0;	i<hidden;	i++) {
				s=0;	w=weight+woff(i,l);
				for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
				q[i]=i?act(wh*s):1;
			}
		}
		o=work+2*layer*hidden;	p=work+aoff(layer-1);
		for(unsigned	i=0;	i<output;	i++) {
			s=0;	w=weight+woff(i,layer);
			for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
			o[i]=wh*s;
		}
		switch(loss) {
		case	0:	for(unsigned	i=0;	i<output;	i++)	if(alpha<0)	y[i]=o[i];	else	o[i]-=y[i];	
			break;
		case	1:	for(unsigned	i=0;	i<output;	i++)	if(alpha<0)	y[i]=1/(1+expf(-o[i]));	else	o[i]=1/(1+expf(-o[i]))-y[i];	
			break;
		case	2:	for(unsigned	i=s=0;	i<output;	i++)	s+=(o[i]=i?expf(o[i]):1);	
					for(unsigned	i=0;	i<output;	i++)	if(alpha<0)	y[i]=o[i]/s;	else	o[i]=i?(o[i]/s-(i==(unsigned)y[0])):0;	
			break;
		}
		if(alpha<0) return;
		for(unsigned	i=0;	i<output;	i++) {
			w=weight+woff(i,layer);	p=work+aoff(layer-1);	g=work+doff(layer-1);	s=o[i]*wh*alpha;
			for(unsigned  j=0;  j<hidden; j++) {	g[j]+=s*w[j];	w[j]-=s*p[j];	}
		}
		for(unsigned	l=layer-1;	l;	l--) {
			p=work+aoff(l-1);	q=work+aoff(l);	g=work+doff(l-1);	h=work+doff(l);
			for(unsigned	i=0;	i<hidden;	i++) {
				w=weight+woff(i,l);	s=h[i]*gra(q[i])*wh;
				for(unsigned  j=0;  j<hidden; j++) {	g[j]+=s*w[j];	w[j]-=s*p[j];	}
			}
		}
		p=work+aoff(0);	g=work+doff(0);
		for(unsigned	i=0;	i<hidden;	i++)	g[i]*=gra(p[i])*wi;
		for(unsigned  i=0;  i<=input; i++) if(wy2u01(wyhash64(i,seed))>=drop) {
			s=(i==input?1:x[i]);	if(s==0)	continue;	w=weight+woff(i,0);
			for(unsigned	j=0;	j<hidden;	j++)	w[j]-=s*g[j];
		}
	}
};
/*	Example:
	wymlp<float,4,16,3,1,0>	model(0.01);
	model.ramdom(time(NULL));
	float	x[4]={1,2,3,5},	y[1]={2};
	model.model(x,y,0.1,wygrand());	//	to learn x-y pair
	model.model(x,y,-1,0);	//	to predict x, and store to y
	model.save("model");
*/