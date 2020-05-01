#include	<sys/mman.h>
#include	<sys/stat.h>
#include	"wyhash32.h"
#include	"float16.h"
#include	<string.h>
#include	<unistd.h>
#include	<float.h>
#include	<stdio.h>
#include	<fcntl.h>
#include	<math.h>
#ifdef	WYMLP_AVX512F
#include	"sgemm.hpp"
#endif
template<unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	output,	unsigned	batch>
class	wymlp{
private:
	int	fd;
	struct	stat	sb;
	float   act(float x){ return  (x/(1+(((int)(x>0)<<1)-1)*x));  }
	float   gra(float x){ return  ((1-(((int)(x>0)<<1)-1)*x)*(1-(((int)(x>0)<<1)-1)*x));  }
	#ifdef WYMLP_RNN
	unsigned	size(void){	return	(input+1)*hidden+hidden*hidden+output*hidden;	}
	unsigned	woff(unsigned	i,	unsigned	l){	return	(l?(l<depth?(input+1)*hidden+i*hidden:(input+1)*hidden+hidden*hidden+i*hidden):i*hidden);	}
	#else
	unsigned	size(void){	return	(input+1)*hidden+(depth-1)*hidden*hidden+output*hidden;	}
	unsigned	woff(unsigned	i,	unsigned	l){	return	 (l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden);	}
	#endif
public:
	float	*weight;
	uint16_t	*weight16;

	wymlp(){	weight=NULL;	weight16=NULL;	}
	void	alloc_weight(void){	free(weight);weight=(float*)aligned_alloc(64,size()*sizeof(float));	}
	void	free_weight(void){	free(weight);	weight=NULL;	}
	void	init_weight(uint64_t	seed){	for(size_t	i=0;	i<size();	i++)	weight[i]=wy2gau(wyrand(&seed));	}
	bool	mmap_weight16(const	char	*F){
		fd=open(F,	O_RDONLY);	if(fd<0)	return	false;
		fstat(fd,	&sb);
		weight16=(uint16_t*)mmap(NULL,	sb.st_size,	PROT_READ,	MAP_SHARED,	fd,	0);
		if(weight16==MAP_FAILED)	return	false;
		return	true;
	}
	void	ummap_weight16(void){	munmap(weight16,sb.st_size);	close(fd);	weight16=NULL;	}
	bool	save(const	char	*F){
		weight16=new	uint16_t[size()];
		for(unsigned	i=0;	i<size();	i++)	weight16[i]=float16(weight[i]);
		FILE	*f=fopen(F,	"wb");
		if(f==NULL)	return	false;
		fwrite(weight16,2ull*size(),1,f);
		fclose(f);
		delete	[]	weight16;	
		weight16=NULL;
		return	true;
	}
	bool	load(const	char	*F){
		if(weight==NULL)	alloc_weight();
		weight16=new	uint16_t[size()];
		FILE	*f=fopen(F,	"rb");
		if(f==NULL)	return	false;
		fread(weight16,2ull*size(),1,f);
		fclose(f);
		for(unsigned	i=0;	i<size();	i++)	weight[i]=float32(weight16[i]);
		delete	[]	weight16;	
		weight16=NULL;
		return	true;
	}
	#ifndef	WYMLP_AVX512F
	void	train(float	**x,	float	**y,	float	eta) {
		const	float	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
		for(unsigned	b=0;	b<batch;	b++){
			float	a[2*depth*hidden+output]={},	*d=a+depth*hidden,	*o=a+2*depth*hidden;
			for(unsigned  i=0;  i<=input; i++)	{
				float	*w=weight+woff(i,0),	s=(i==input?1:x[b][i]);
				for(unsigned	j=0;	j<hidden;	j++)	a[j]+=s*w[j];
			}
			for(unsigned	i=0;	i<hidden;	i++) a[i]=i?act(wi*a[i]):1;
			for(unsigned	l=1;	l<=depth;	l++) {
				float	*p=a+(l-1)*hidden,	*q=(l==depth?o:a+l*hidden);
				for(unsigned	i=0;	i<(l==depth?output:hidden);	i++) {
					float	*w=weight+woff(i,l),	s=0;
					for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
					q[i]=(l==depth?s*wh:(i?act(s*wh):1));
				}
			}
			for(unsigned	i=0;	i<output;	i++)	o[i]=(o[i]-y[b][i])*eta;
			for(unsigned	l=depth;	l;	l--) {
				float	*p=a+(l-1)*hidden,	*q=(l==depth?o:a+l*hidden),	*g=d+(l-1)*hidden,	*h=(l==depth?o:d+l*hidden);
				for(unsigned	i=0;	i<(l==depth?output:hidden);	i++) {
					float	*w=weight+woff(i,l),	s=(l==depth?q[i]:h[i]*gra(q[i]))*wh;
					for(unsigned  j=0;  j<hidden; j++) {	g[j]+=s*w[j];	w[j]-=s*p[j];	}
				}
			}
			for(unsigned	i=0;	i<hidden;	i++) d[i]*=gra(a[i])*wi;
			for(unsigned  i=0;  i<=input; i++)	{
				float	*w=weight+woff(i,0),	s=(i==input?1:x[b][i]);
				for(unsigned	j=0;	j<hidden;	j++)	w[j]-=s*d[j];
			}
		}
	}
	#else
	void	train(float	**x,	float	**y,	float	eta) {
		#define	aoff(b,l)	(a+(l)*batch*hidden+(b)*hidden)
		#define	doff(b,l)	(d+(l)*batch*hidden+(b)*hidden)
		const	float	wh=1/sqrtf(hidden), wi=1/sqrtf(input+1);
		float   *a=(float*)aligned_alloc(64,(2*depth*batch*hidden+batch*output)*sizeof(float));	
		float	*d=a+depth*batch*hidden,	*o=d+depth*batch*hidden,	*p,	*q,	*w;
		memset(a,	0,	batch*hidden*sizeof(float));
		for(unsigned	b=0;	b<batch;	b++){
			p=aoff(b,0);
			for(unsigned	i=0;	i<input;	i++){
				w=weight+woff(i,0);	float	s=x[b][i];
				for(unsigned	j=0;	j<hidden;	j++)	p[j]+=s*w[j];
			}
			w=weight+woff(input,0);	p[0]=1;
			for(unsigned	j=1;	j<hidden;	j++)	p[j]=act(wi*(p[j]+w[j]));
		}
		for(unsigned	l=1;	l<depth;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,0>(wh,weight+woff(0,l),aoff(0,l-1),aoff(0,l));
			for(unsigned    b=0;    b<batch;    b++){
				p=aoff(b,l);	p[0]=1;
				for(unsigned	j=1;	j<hidden;	j++)	p[j]=act(p[j]);
			}
		}
		sgemm<1,0,output,batch,hidden,hidden,hidden,output,0>(wh,weight+woff(0,depth),aoff(0,depth-1),o);
		for(unsigned    b=0;    b<batch;    b++){
			p=o+b*output;	
			for(unsigned	i=0;	i<output;	i++)	p[i]=(p[i]-y[b][i])*eta*wh;
		}
		sgemm<0,0,hidden,batch,output,hidden,output,hidden,0>(1,weight+woff(0,depth),o,doff(0,depth-1));
		sgemm<0,1,hidden,output,batch,hidden,output,hidden,1>(-1,aoff(0,depth-1),o,weight+woff(0,depth));
		for(unsigned	l=depth-1;	l;	l--) {
			for(unsigned	b=0;	b<batch;	b++){
				p=aoff(b,l);	q=doff(b,l);
				for(unsigned	i=0;	i<hidden;	i++)	q[i]*=gra(p[i])*wh;
			}
			sgemm<0,0,hidden,batch,hidden,hidden,hidden,hidden,0>(1,weight+woff(0,l),doff(0,l),doff(0,l-1));
			sgemm<0,1,hidden,hidden,batch,hidden,hidden,hidden,1>(-1,aoff(0,l-1),doff(0,l),weight+woff(0,l));
		}
		for(unsigned    b=0;    b<batch;    b++){
			w=weight+woff(input,0);	p=aoff(b,0);	q=doff(b,0);
			for(unsigned	j=0;	j<hidden;	j++){	q[j]*=gra(p[j])*wi;	w[j]-=q[j];	}
			for(unsigned	i=0;	i<input;	i++){
				w=weight+woff(i,0);	float	s=x[b][i];
				for(unsigned	j=0;	j<hidden;	j++)	w[j]-=s*q[j];
			}
		}
		free(a);
	}
	#endif
	void	predict(float	*x,	float	*y) {
		float	a[depth*hidden+output]= {},	*o=a+depth*hidden,	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
		for(unsigned  i=0;  i<=input; i++)	{
			float	*w=weight+woff(i,0),	s=(i==input?1:x[i]);
			for(unsigned	j=0;	j<hidden;	j++)	a[j]+=s*w[j];
		}
		for(unsigned	i=0;	i<hidden;	i++) a[i]=i?act(wi*a[i]):1;
		for(unsigned	l=1;	l<=depth;	l++) {
			float	*p=a+(l-1)*hidden,	*q=(l==depth?o:a+l*hidden);
			for(unsigned	i=0;	i<(l==depth?output:hidden);	i++) {
				float	*w=weight+woff(i,l),	s=0;
				for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
				q[i]=(l==depth?s*wh:(i?act(s*wh):1));
			}
		}
		for(unsigned    i=0;    i<output;   i++)	y[i]=o[i];
	}
	void	predict16(float	*x,	float	*y) {
		float	a[depth*hidden+output]= {},	*o=a+depth*hidden,	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1),	v;
		for(unsigned  i=0;  i<=input; i++)	{
			uint16_t	*w=weight16+woff(i,0);
			float	s=(i==input?1:x[i]);
			for(unsigned	j=0;	j<hidden;	j++){	v=float32(w[j]);	a[j]+=s*v;	}
		}
		for(unsigned	i=0;	i<hidden;	i++) a[i]=i?act(wi*a[i]):1;
		for(unsigned	l=1;	l<=depth;	l++) {
			float	*p=a+(l-1)*hidden,	*q=(l==depth?o:a+l*hidden);
			for(unsigned	i=0;	i<(l==depth?output:hidden);	i++) {
				uint16_t	*w=weight+woff(i,l);
				float	s=0;
				for(unsigned	j=0;	j<hidden;	j++){	v=float32(w[j]);	s+=v*p[j];	}
				q[i]=(l==depth?s*wh:(i?act(s*wh):1));
			}
		}
		for(unsigned    i=0;    i<output;   i++)	y[i]=o[i];
	}
};
