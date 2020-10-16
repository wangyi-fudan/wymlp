#include	<math.h>
static	inline	float	tlfn_acti(float	x) {	return  x/(1+fabsf(x));	}
static	inline	float	tlfn_grad(float	x) {	x=1-fabsf(x);	return	x*x;	}
static	inline	unsigned	tlfn_size(unsigned	input,	unsigned	hidden,	unsigned	output){	return	(input+1)*hidden+hidden*hidden+output*hidden;	}
template<unsigned	hidden,	unsigned	output>
void	tlfn(unsigned	input,	float	*weight,	float	*x,	float	*y,	float	eta) {
	float	a[4*hidden+output]={},	*a1=a+hidden,	*d=a1+hidden,	*d1=d+hidden,	*o=d1+hidden,	s,	*w,	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
	unsigned	i,	j;
	for(i=0;	i<=input;	i++){
		s=i<input?x[i]:1;	w=weight+i*hidden;
		for(j=0;	j<hidden;	j++)	a[j]+=s*w[j];
	}
	for(i=0;	i<hidden;	i++){	a[i]=tlfn_acti(wi*a[i]);	}	a[0]=1;
	for(i=1;	i<hidden;	i++){
		s=0;	w=weight+(input+1)*hidden+i*hidden;
		for(j=0;	j<hidden;	j++)	s+=a[j]*w[j];
		a1[i]=s;
	}
	for(i=0;	i<hidden;	i++){	a1[i]=tlfn_acti(a1[i]*wh);	}	a1[0]=1;
	for(i=0;	i<output;	i++){
		s=0;	w=weight+(input+1)*hidden+hidden*hidden+i*hidden;
		for(j=0;	j<hidden;	j++)	s+=w[j]*a1[j];
		o[i]=s*wh;
	}
	if(eta<0){	for(i=0;	i<output;	i++){	y[i]=o[i];	}	return;	}
	for(i=0;	i<output;	i++){
		s=(o[i]>y[i]?1:-1)*wh*eta;	w=weight+(input+1)*hidden+hidden*hidden+i*hidden;
		for(j=0;	j<hidden;	j++){	d1[j]+=s*w[j];	w[j]-=s*a1[j];	}
	}
	for(i=0;	i<hidden;	i++){
		s=d1[i]*tlfn_grad(a1[i])*wh;	w=weight+(input+1)*hidden+i*hidden;
		for(j=0;	j<hidden;	j++){	d[j]+=s*w[j];	w[j]-=s*a[j];	}
	}
	for(i=0;	i<hidden;	i++)	d[i]*=tlfn_grad(a[i])*wi;
	for(i=0;	i<=input;	i++){
		s=i<input?x[i]:1;	w=weight+i*hidden;
		for(j=0;	j<hidden;	j++)	w[j]-=s*d[j];
	}
}
