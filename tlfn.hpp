#include	<math.h>
template<unsigned	input,	unsigned	hidden,	unsigned	output>
struct	tlfn {
	float	act(float	x) {	return  x/sqrtf(1+x*x);	}
	float	gra(float	x) {	x=1-x*x;	return	x*sqrtf(x);	}
	float	weight[(input+1)*hidden+hidden*hidden+output*hidden];
	void	model(float	*x,	float	*y,	float	eta) {
		const	float	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);	unsigned	i,	j;
		float	a[4*hidden+output]={},	*a1=a+hidden,	*d=a1+hidden,	*d1=d+hidden,	*o=d1+hidden,	s,	*w;
		for(i=0;	i<=input;	i++){
			s=i<input?x[i]:1;	w=weight+i*hidden;
			for(j=0;	j<hidden;	j++)	a[j]+=s*w[j];
		}
		for(i=0;	i<hidden;	i++){	a[i]=act(wi*a[i]);	}	a[0]=1;
		for(i=1;	i<hidden;	i++){
			s=0;	w=weight+(input+1)*hidden+i*hidden;
			for(j=0;	j<hidden;	j++)	s+=a[j]*w[j];
			a1[i]=s;
		}
		for(i=0;	i<hidden;	i++){	a1[i]=act(a1[i]*wh);	}	a1[0]=1;
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
			s=d1[i]*gra(a1[i])*wh;	w=weight+(input+1)*hidden+i*hidden;
			for(j=0;	j<hidden;	j++){	d[j]+=s*w[j];	w[j]-=s*a[j];	}
		}
		for(i=0;	i<hidden;	i++)	d[i]*=gra(a[i])*wi;
		for(i=0;	i<=input;	i++){
			s=i<input?x[i]:1;	w=weight+i*hidden;
			for(j=0;	j<hidden;	j++)	w[j]-=s*d[j];
		}
	}
};
