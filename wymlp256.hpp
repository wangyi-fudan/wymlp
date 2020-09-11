#include	"sgemm256.hpp"
#include	<sys/mman.h>
#include	<sys/stat.h>
#include	"wyhash.h"
#include	<string.h>
#include	<unistd.h>
#include	<float.h>
#include	<stdio.h>
#include	<fcntl.h>
#include	<math.h>
#include	<omp.h>
template<unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	output,	unsigned	batch>
class	wymlp{
private:
	int	fd;
	struct	stat	sb;
	float	act(float	x){	return	x>1?1:(x<-1?-1:x);	}
	float	gra(float	x){	return	x>=1||x<=-1?0:1;	}
	unsigned	woff(unsigned	i,	unsigned	l){	return	 (l?input*hidden+(l==depth)*hidden*hidden+i*hidden:i*input);	}
public:
	float	*weight;
	wymlp(){	weight=NULL;	}
	const	uint64_t	size(void){	return	input*hidden+hidden*hidden+output*hidden;	}
	void	alloc_weight(void){	free(weight);	weight=(float*)aligned_alloc(64,size()*sizeof(float));	}
	void	free_weight(void){	free(weight);	weight=NULL;	}
	void	init_weight(uint64_t	seed){	for(size_t	i=0;	i<size();	i++)	weight[i]=1.592537420f*wy2gau(wyrand(&seed));	}
	bool	mmap_weight(const	char	*F){
		fd=open(F,	O_RDONLY);	if(fd<0)	return	false;
		fstat(fd,	&sb);
		weight=(float*)mmap(NULL,	sb.st_size,	PROT_READ,	MAP_SHARED,	fd,	0);
		if(weight==MAP_FAILED)	return	false;
		return	true;
	}
	void	munmap_weight(void){	munmap(weight,sb.st_size);	close(fd);	weight=NULL;	}
	bool	save(const	char	*F){
		FILE	*f=fopen(F,	"wb");
		if(f==NULL)	return	false;
		fwrite(weight,sizeof(float),size(),f);
		fclose(f);
		return	true;
	}
	bool	load(const	char	*F){
		if(weight==NULL)	alloc_weight();
		FILE	*f=fopen(F,	"rb");
		if(f==NULL)	return	false;
		fread(weight,sizeof(float),size(),f);
		fclose(f);
		return	true;
	}
	float	train(float	x[batch][2],	float	y[batch][3],	float	eta) {
		#define	aoff(b,l)	(a+(l)*batch*hidden+(b)*hidden)
		#define	doff(b,l)	(d+(l)*batch*hidden+(b)*hidden)
		const	float	wh=1/sqrtf(hidden), wi=1/sqrtf(input);
		float   *a=(float*)aligned_alloc(64,(2*depth*batch*hidden+batch*output)*sizeof(float));	
		float	*d=a+depth*batch*hidden,	*o=d+depth*batch*hidden,	*p,	*q,	ret=0;
		float	inp[batch*input];
		for(unsigned	b=0;	b<batch;	b++){
			p=inp+b*input;
			for(unsigned	i=0;	i<input-1;	i++)	p[i]=x[b][i];
			p[input-1]=1;
		}	
		sgemm<1,0,hidden,batch,input,input,input,hidden,0>(wi,weight,inp,a);
		for(unsigned	b=0;	b<batch;	b++){
			p=aoff(b,0);
			for(unsigned	i=0;	i<hidden;	i++)	p[i]=act(p[i]);
			p[0]=1;
		}
		for(unsigned	l=1;	l<depth;	l++){
			sgemm<1,0,hidden,batch,hidden,hidden,hidden,hidden,0>(wh,weight+woff(0,l),aoff(0,l-1),aoff(0,l));
			for(unsigned    b=0;    b<batch;    b++){
				p=aoff(b,l);
				for(unsigned	i=0;	i<hidden;	i++)	p[i]=act(p[i]);
				p[0]=1;
			}
		}
		sgemm<1,0,output,batch,hidden,hidden,hidden,output,0>(wh,weight+woff(0,depth),aoff(0,depth-1),o);
		for(unsigned    b=0;    b<batch;    b++){
			p=o+b*output;
			for(unsigned	i=0;	i<output;	i++){
				p[i]=1/(1+expf(-p[i]));
				if(eta<0)	y[b][i]=p[i];
				else{
					ret-=y[b][i]*logf(fmaxf(p[i],FLT_MIN))+(1-y[b][i])*logf(fmaxf(1-p[i],FLT_MIN));
					p[i]=(p[i]-y[b][i])*wh*eta;
				}
			}
		}
		if(eta<0)	return	0;
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
			p=aoff(b,0);	q=doff(b,0);
			for(unsigned	j=0;	j<hidden;	j++)	q[j]*=gra(p[j])*wi;
		}
		sgemm<0,1,input,hidden,batch,input,hidden,input,1>(-1,inp,d,weight);
		free(a);
		return	ret;
	}
};
