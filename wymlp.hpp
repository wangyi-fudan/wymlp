#include	<math.h>
template<unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	output>
struct	wymlp {
	float	act(float	x) {	return  x/(1+fabsf(x));	}
	float	gra(float	x) {	x=1-fabsf(x);	return	x*x;	}
	#define	wymlp_size	((input+1)*hidden+depth*hidden*hidden+output*hidden)
	float	weight[(input+1)*hidden+depth*hidden*hidden+output*hidden];
	void	model(float	*x,	float	*y,	float	eta) {
		const	float	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
		float	a[depth][hidden]={},	d[depth][hidden]={},	o[output];
		for(unsigned	i=0;	i<=input;	i++){
			float	s=i<input?x[i]:1,	*w=weight+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++)	a[0][j]+=s*w[j];
		}
		for(unsigned	i=0;	i<hidden;	i++){	a[0][i]=act(wi*a[0][i]);	}	a[0][0]=1;
		for(unsigned	l=1;	l<depth;	l++){
			for(unsigned	i=1;	i<hidden;	i++){
				float	s=0,	*w=weight+(input+1)*hidden+(l-1)*hidden*hidden+i*hidden;
				for(unsigned	j=0;	j<hidden;	j++)	s+=a[l-1][j]*w[j];
				a[l][i]=s;
			}
			for(unsigned	i=0;	i<hidden;	i++){	a[l][i]=act(a[l][i]*wh);	}	a[l][0]=1;
		}
		for(unsigned	i=0;	i<output;	i++){
			float	s=0,	*w=weight+(input+1)*hidden+(depth-1)*hidden*hidden+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*a[depth-1][j];
			o[i]=s*wh;
		}
		if(eta<0){	for(unsigned	i=0;	i<output;	i++){	y[i]=o[i];	}	return;	}
		for(unsigned	i=0;	i<output;	i++){
			float	s=(o[i]>y[i]?1:-1)*wh*eta,	*w=weight+(input+1)*hidden+(depth-1)*hidden*hidden+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++){	d[depth-1][j]+=s*w[j];	w[j]-=s*a[depth-1][j];	}
		}
		for(unsigned	l=depth-1;	l;	l--){
			for(unsigned	i=0;	i<hidden;	i++){
				float	s=d[l][i]*gra(a[l][i])*wh,	*w=weight+(input+1)*hidden+(l-1)*hidden*hidden+i*hidden;
				for(unsigned	j=0;	j<hidden;	j++){	d[l-1][j]+=s*w[j];	w[j]-=s*a[l-1][j];	}
			}
		}
		for(unsigned	i=0;	i<hidden;	i++)	d[0][i]*=gra(a[0][i])*wi;
		for(unsigned	i=0;	i<=input;	i++){
			float	s=i<input?x[i]:1,	*w=weight+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++)	w[j]-=s*d[0][j];
		}
	}
};
