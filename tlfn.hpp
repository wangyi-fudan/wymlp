#include	<stdio.h>
#include	<math.h>
template<unsigned	input,	unsigned	hidden,	unsigned	output>
class	tlfn{
private:
	float	acti(float	x) {	return  x/(1+(x>0?x:-x));	}
	float	grad(float	x) {	return	x=1-(x>0?x:-x);	return	x*x;	}
public:
	float	weight[(input+1)*hidden+hidden*hidden+output*hidden];
	void	init(uint64_t	&seed){	for(unsigned	i=0;	i<sizeof(weight)/sizeof(float);	i++)	weight[i]=sqrtf(1+(i>=(input+1)*hidden))*wy2gau(wyrand(&seed));	}
	
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
		for(unsigned	i=0;	i<hidden;	i+=4){
			float	s0=0,	*w0=weight+(input+1)*hidden+i*hidden;
			float	s1=0,	*w1=w0+hidden;
			float	s2=0,	*w2=w1+hidden;
			float	s3=0,	*w3=w2+hidden;
			for(unsigned	j=0;	j<hidden;	j++){	s0+=a[j]*w0[j];	s1+=a[j]*w1[j];	s2+=a[j]*w2[j];	s3+=a[j]*w3[j];	}
			a1[i]=s0;	a1[i+1]=s1;	a1[i+2]=s2;	a1[i+3]=s3;
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
		d1[0]=0;
		for(unsigned	i=0;	i<hidden;	i+=4){
			float	s0=d1[i]*grad(a1[i])*wh,	*w0=weight+(input+1)*hidden+i*hidden;
			float	s1=d1[i+1]*grad(a1[i+1])*wh,	*w1=w0+hidden;
			float	s2=d1[i+2]*grad(a1[i+2])*wh,	*w2=w1+hidden;
			float	s3=d1[i+3]*grad(a1[i+3])*wh,	*w3=w2+hidden;
			for(unsigned	j=0;	j<hidden;	j++){	
				d[j]+=s0*w0[j]+s1*w1[j]+s2*w2[j]+s3*w3[j];	
				w0[j]-=s0*a[j];	w1[j]-=s1*a[j];	w2[j]-=s2*a[j];	w3[j]-=s3*a[j];	
			}
		}
		for(unsigned	i=0;	i<hidden;	i++)	d[i]*=grad(a[i])*wi;
		d[0]=0;
		for(unsigned	i=0;	i<=input;	i++){
			float	s=i<input?x[i]:1,	*w=weight+i*hidden;
			for(unsigned	j=0;	j<hidden;	j++)	w[j]-=s*d[j];
		}
	}
};
