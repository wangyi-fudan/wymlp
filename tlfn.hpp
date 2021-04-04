#include	"wyhash.h"
#include	<stdio.h>
#include	<math.h>
template<uint32_t	input,	uint32_t	hidden,	uint32_t	output>
class	tlfn{
private:
	float	acti(float	x) {	return  x/(1+(x>0?x:-x));	}
	float	grad(float	x) {	x=1-(x>0?x:-x);	return	x*x;	}
public:
	float	weight[(input+1)*hidden+hidden*hidden+output*hidden];

	tlfn(uint64_t	seed){	for(uint32_t	i=0;	i<sizeof(weight)/sizeof(float);	i++)	weight[i]=wy2gau(wyrand(&seed));	}

	bool	save(const	char	*F){
		FILE	*f=fopen(F,	"wb");
		if(f==NULL)	return	false;
		uint32_t	n;
		n=input;	fwrite(&n,4,1,f);
		n=hidden;	fwrite(&n,4,1,f);
		n=output;	fwrite(&n,4,1,f);
		fwrite(weight,sizeof(weight),1,f);
		fclose(f);
		return	true;
	}

	bool	load(const	char	*F){
		FILE	*f=fopen(F,	"rb");
		if(f==NULL)	return	false;
		uint32_t	n;
		if(fread(&n,4,1,f)!=1||n!=input)	return	false;
		if(fread(&n,4,1,f)!=1||n!=hidden)	return	false;
		if(fread(&n,4,1,f)!=1||n!=output)	return	false;
		if(fread(weight,sizeof(weight),1,f)!=1)	return	false;
		fclose(f);
		return	true;
	}

	void	operator()(float	*x,	float	*y,	float	eta) {
		float	a[4*hidden+output]={},	*a1=a+hidden,	*d=a1+hidden,	*d1=d+hidden,	*o=d1+hidden;
		const	float	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
		for(uint32_t	i=0;	i<=input;	i++){
			float	s=i<input?x[i]:1,	*w=weight+i*hidden;
			for(uint32_t	j=0;	j<hidden;	j++)	a[j]+=s*w[j];
		}
		for(uint32_t	i=0;	i<hidden;	i++){	a[i]=acti(wi*a[i]);	}	a[0]=1;
		for(uint32_t	i=0;	i<hidden;	i+=4){
			float	s=0,	*w=weight+(input+1)*hidden+i*hidden;
			for(uint32_t	j=0;	j<hidden;	j++)	s+=a[j]*w[j];
			a1[i]=s;
		}
		for(uint32_t	i=0;	i<hidden;	i++){	a1[i]=acti(a1[i]*wh);	}	a1[0]=1;
		for(uint32_t	i=0;	i<output;	i++){
			float	s=0,	*w=weight+(input+1)*hidden+hidden*hidden+i*hidden;
			for(uint32_t	j=0;	j<hidden;	j++)	s+=w[j]*a1[j];
			o[i]=s*wh;
		}
		if(eta<0){	for(uint32_t	i=0;	i<output;	i++){	y[i]=o[i];	}	return;	}
		for(uint32_t	i=0;	i<output;	i++){
			float	s=fminf(fmaxf(o[i]-y[i],-1),1)*wh*eta,	*w=weight+(input+1)*hidden+hidden*hidden+i*hidden;
			for(uint32_t	j=0;	j<hidden;	j++){	d1[j]+=s*w[j];	w[j]-=s*a1[j];	}
		}	d1[0]=0;
		for(uint32_t	i=0;	i<hidden;	i++){
			float	s=d1[i]*grad(a1[i])*wh,	*w=weight+(input+1)*hidden+i*hidden;
			for(uint32_t	j=0;	j<hidden;	j++){	d[j]+=s*w[j];	w[j]-=s*a[j];	}
		}
		for(uint32_t	i=0;	i<hidden;	i++){	d[i]*=grad(a[i])*wi;	}	d[0]=0;
		for(uint32_t	i=0;	i<=input;	i++){
			float	s=i<input?x[i]:1,	*w=weight+i*hidden;
			for(uint32_t	j=0;	j<hidden;	j++)	w[j]-=s*d[j];
		}
	}
};
