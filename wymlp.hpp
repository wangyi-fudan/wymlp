#include	<math.h>
static	inline	float	wymlp_activate(float	x) {	return  x/(1+fabsf(x));	}
static	inline	float	wymlp_gradient(float	x) {	x=1-fabsf(x);	return	x*x;	}
static	inline	unsigned	wymlp_size(unsigned   input,	unsigned	hidden,	unsigned	depth,	unsigned	output){
	return	(input+1)*hidden+depth*hidden*hidden+output*hidden;
}
template<unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	output>
static	inline	void	wymlp(float	*weight,	float	*x,	float	*y,	float	eta){
	float	a[2*depth*hidden+output]={},*o=a+2*depth*hidden,wh=1/sqrtf(hidden),wi=1/sqrtf(input+1),s,*w,*p,*q,*g,*h;
	unsigned	i,	j,	l;
	for(i=0;	i<=input;	i++){
		s=i<input?x[i]:1;	w=weight+i*hidden;
		for(j=0;	j<hidden;	j++)	a[j]+=s*w[j];
	}
	for(i=0;	i<hidden;	i++){	a[i]=wymlp_activate(wi*a[i]);	}	a[0]=1;
	for(l=1;	l<depth;	l++){
		p=a+(l-1)*hidden;	q=p+hidden;
		for(i=1;	i<hidden;	i++){
			s=0;	w=weight+(input+1)*hidden+(l-1)*hidden*hidden+i*hidden;
			for(j=0;	j<hidden;	j++)	s+=p[j]*w[j];
			q[i]=s;
		}
		for(i=0;	i<hidden;	i++){	q[i]=wymlp_activate(q[i]*wh);	}	q[0]=1;
	}
	p=a+(depth-1)*hidden;	g=a+(depth+depth-1)*hidden;
	for(i=0;	i<output;	i++){
		s=0;	w=weight+(input+1)*hidden+(depth-1)*hidden*hidden+i*hidden;
		for(j=0;	j<hidden;	j++)	s+=w[j]*p[j];
		o[i]=s*wh;
	}
	if(eta<0){	for(i=0;	i<output;	i++){	y[i]=o[i];	}	return;	}
	for(i=0;	i<output;	i++){
		s=(o[i]>y[i]?1:-1)*wh*eta;	w=weight+(input+1)*hidden+(depth-1)*hidden*hidden+i*hidden;
		for(j=0;	j<hidden;	j++){	g[j]+=s*w[j];	w[j]-=s*p[j];	}
	}
	for(l=depth-1;	l;	l--){
		p=a+(l-1)*hidden;	q=p+hidden;	g=a+(depth+l-1)*hidden;	h=g+hidden;
		for(i=0;	i<hidden;	i++){
			s=h[i]*wymlp_gradient(q[i])*wh;	w=weight+(input+1)*hidden+(l-1)*hidden*hidden+i*hidden;
			for(j=0;	j<hidden;	j++){	g[j]+=s*w[j];	w[j]-=s*p[j];	}
		}
	}
	g=a+depth*hidden;
	for(i=0;	i<hidden;	i++)	g[i]*=wymlp_gradient(a[i])*wi;
	for(i=0;	i<=input;	i++){
		s=i<input?x[i]:1;	w=weight+i*hidden;
		for(j=0;	j<hidden;	j++)	w[j]-=s*g[j];
	}
};
