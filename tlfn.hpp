#include	<stdio.h>
#include	<math.h>
template<unsigned	input,	unsigned	hidden,	unsigned	output>
class	tlfn{
private:
	float	acti(float	x) {	return  x/(1+fabsf(x));	}
	float	grad(float	x) {	x=1-fabsf(x);	return	x*x;	}
public:
	float	weight[(input+1)*hidden+hidden*hidden+output*hidden];
	void	init(uint64_t	&seed){	for(unsigned	i=0;	i<sizeof(weight)/sizeof(float);	i++)	weight[i]=(1+(i>=(input+1)*hidden))*wy2gau(wyrand(&seed));	}
	
	bool	save(const	char	*F){
		FILE	*f=fopen(F,	"wb");
		if(f==NULL)	return	false;
		unsigned	n;
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
		unsigned	n;
		if(fread(&n,4,1,f)!=1||n!=input)	return	false;
		if(fread(&n,4,1,f)!=1||n!=hidden)	return	false;
		if(fread(&n,4,1,f)!=1||n!=output)	return	false;
		if(fread(weight,sizeof(weight),1,f)!=1)	return	false;
		fclose(f);
		return	true;
	}

	void	model(float	*x,	float	*y,	float	eta) {
		float	a[4*hidden+output]={},	*a1=a+hidden,	*d=a1+hidden,	*d1=d+hidden,	*o=d1+hidden,	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
		for(unsigned	i=0;	i<=input;	i++){
			float	s=i<input?x[i]:1,	*w=weight+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++)	a[j]+=s*w[j];
		}
		for(unsigned	i=0;	i<hidden;	i++){	a[i]=acti(wi*a[i]);	}	a[0]=1;
		for(unsigned	i=1;	i<hidden;	i++){
			float	s=0,	*w=weight+(input+1)*hidden+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++)	s+=a[j]*w[j];
			a1[i]=s;
		}
		for(unsigned	i=0;	i<hidden;	i++){	a1[i]=acti(a1[i]*wh);	}	a1[0]=1;
		for(unsigned	i=0;	i<output;	i++){
			float	s=0,	*w=weight+(input+1)*hidden+hidden*hidden+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*a1[j];
			o[i]=s*wh;
		}
		if(eta<0){	for(unsigned	i=0;	i<output;	i++){	y[i]=o[i];	}	return;	}
		for(unsigned	i=0;	i<output;	i++){
			float	s=(o[i]>y[i]?1:-1)*wh*eta,	*w=weight+(input+1)*hidden+hidden*hidden+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++){	d1[j]+=s*w[j];	w[j]-=s*a1[j];	}
		}
		for(unsigned	i=0;	i<hidden;	i++){
			float	s=d1[i]*grad(a1[i])*wh,	*w=weight+(input+1)*hidden+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++){	d[j]+=s*w[j];	w[j]-=s*a[j];	}
		}
		for(unsigned	i=0;	i<hidden;	i++)	d[i]*=grad(a[i])*wi;
		for(unsigned	i=0;	i<=input;	i++){
			float	s=i<input?x[i]:1,	*w=weight+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++)	w[j]-=s*d[j];
		}
	}
};
