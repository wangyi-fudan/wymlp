#include	<stdlib.h>
#include	<stdio.h>
#include	<math.h>
template<size_t	input,	size_t	hidden,	size_t	output>
struct	tlfn{
	float	acti(float	x) {	return  x/sqrtf(1+x*x);	}
	float	grad(float	x) {	x=1-x*x;	return	x*sqrtf(x);	}
	float	*weight;
	tlfn(){	
		size_t	size=(input+1)*hidden+hidden*hidden+output*hidden;
		weight=(float*)aligned_alloc(64,size*sizeof(float));
		for(size_t	i=0;	i<size;	i++)	weight[i]=2.0f*rand()/(float)RAND_MAX-1;
	}
	~tlfn(){	free(weight);	}
	void	operator()(float	*x,	float	*y,	float	eta) {//eta<0 for prediction stored in y
		float	a[4*hidden+output]={},	*a1=a+hidden,	*d=a1+hidden,	*d1=d+hidden,	*o=d1+hidden;
		const	float	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
		for(size_t	i=0;	i<=input;	i++){
			float	s=i<input?x[i]:1,	*w=weight+i*hidden;
			for(size_t	j=0;	j<hidden;	j++)	a[j]+=s*w[j];
		}
		for(size_t	i=0;	i<hidden;	i++){	a[i]=acti(wi*a[i]);	}	a[0]=1;
		for(size_t	i=0;	i<hidden;	i++){
			float	s=0,	*w=weight+(input+1)*hidden+i*hidden;
			for(size_t	j=0;	j<hidden;	j++)	s+=a[j]*w[j];
			a1[i]=s;
		}
		for(size_t	i=0;	i<hidden;	i++){	a1[i]=acti(a1[i]*wh);	}	a1[0]=1;
		for(size_t	i=0;	i<output;	i++){
			float	s=0,	*w=weight+(input+1)*hidden+hidden*hidden+i*hidden;
			for(size_t	j=0;	j<hidden;	j++)	s+=w[j]*a1[j];
			o[i]=s*wh;
		}
		if(eta<0){	for(size_t	i=0;	i<output;	i++){	y[i]=o[i];	}	return;	}
		for(size_t	i=0;	i<output;	i++){
			float	s=(o[i]-y[i])*wh*eta,	*w=weight+(input+1)*hidden+hidden*hidden+i*hidden;
			for(size_t	j=0;	j<hidden;	j++){	d1[j]+=s*w[j];	w[j]-=s*a1[j];	}
		}d1[0]=0;
		for(size_t	i=0;	i<hidden;	i++){
			float	s=d1[i]*grad(a1[i])*wh,	*w=weight+(input+1)*hidden+i*hidden;
			for(size_t	j=0;	j<hidden;	j++){	d[j]+=s*w[j];	w[j]-=s*a[j];	}
		}
		for(size_t	i=0;	i<hidden;	i++){	d[i]*=grad(a[i])*wi;	}	d[0]=0;
		for(size_t	i=0;	i<=input;	i++){
			float	s=i<input?x[i]:1,	*w=weight+i*hidden;
			for(size_t	j=0;	j<hidden;	j++)	w[j]-=s*d[j];
		}
	}
};
