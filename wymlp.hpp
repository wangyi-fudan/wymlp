#include	<sys/mman.h>
#include	<sys/stat.h>
#include	"wyhash32.h"
#include	<string.h>
#include	<unistd.h>
#include	<stdio.h>
#include	<fcntl.h>
#include	<math.h>
#include	<omp.h>
template<class	type,	unsigned	input,	unsigned	hidden,	unsigned	depth,	unsigned	output,	unsigned	threads>
class	wymlp{
private:
	omp_lock_t	lock;
	int	fd;
	struct	stat	sb;
	type	*gradient;
	unsigned	accu[threads]={};
	type	act(type	x){ return	x/sqrt(1+x*x);	}
	type	gra(type	x){	x=1-x*x;	return	x*sqrt(x);	}
	uint64_t	size(void){	return	(input+1)*hidden+(depth-1)*hidden*hidden+output*hidden;	}
	unsigned	woff(unsigned	i,	unsigned	l){	return	 (l?(input+1)*hidden+(l-1)*hidden*hidden+i*hidden:i*hidden);	}
public:
	type	*weight;
	wymlp(){	weight=NULL;	gradient=NULL;	omp_set_num_threads(threads);	omp_init_lock(&lock);	}
	~wymlp(){	omp_destroy_lock(&lock);	}
	void	alloc_weight(void){	
		free(weight);	weight=(type*)aligned_alloc(64,size()*sizeof(type));	
		free(gradient);	gradient=(type*)aligned_alloc(64,threads*size()*sizeof(type));	
		memset(gradient,0,threads*size()*sizeof(type));
	}
	void	free_weight(void){
		free(weight);	weight=NULL;	
		free(gradient);	gradient=NULL;	
	}
	void	init_weight(uint64_t	seed){	for(size_t	i=0;	i<size();	i++)	weight[i]=wy2gau(wyrand(&seed));	}
	bool	mmap_weight(const	char	*F){
		fd=open(F,	O_RDONLY);	if(fd<0)	return	false;
		fstat(fd,	&sb);
		weight=(type*)mmap(NULL,	sb.st_size,	PROT_READ,	MAP_SHARED,	fd,	0);
		if(weight==MAP_FAILED)	return	false;
		return	true;
	}
	void	munmap_weight(void){	munmap(weight,sb.st_size);	close(fd);	weight=NULL;	}
	bool	save(const	char	*F){
		FILE	*f=fopen(F,	"wb");
		if(f==NULL)	return	false;
		fwrite(weight,sizeof(type),size(),f);
		fclose(f);
		return	true;
	}
	bool	load(const	char	*F){
		if(weight==NULL)	alloc_weight();
		FILE	*f=fopen(F,	"rb");
		if(f==NULL)	return	false;
		fread(weight,sizeof(type),size(),f);
		fclose(f);
		return	true;
	}
	void	train(float	*x,	float	*y,	type	eta,	unsigned	tid) {
		const	type	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
		type	a[2*depth*hidden+output]={},	*d=a+depth*hidden,	*o=d+depth*hidden,	*w,	s,	*grd=gradient+tid*size(),	*gr;
		for(unsigned  i=0;  i<input; i++)	{
			w=weight+woff(i,0);	s=x[i];
			for(unsigned	j=0;	j<hidden;	j++)	a[j]+=s*w[j];
		}
		w=weight+woff(input,0);
		for(unsigned	i=0;	i<hidden;	i++) a[i]=act(wi*(a[i]+w[i]));
		a[0]=1;
		for(unsigned	l=1;	l<depth;	l++) {
			type	*p=a+(l-1)*hidden,	*q=a+l*hidden;
			for(unsigned	i=0;	i<hidden;	i++) {
				w=weight+woff(i,l);	s=0;
				for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
				q[i]=act(s*wh);
			}
			q[0]=1;
		}
		{
			type	*p=a+(depth-1)*hidden;	eta*=wh;
			for(unsigned	i=0;	i<output;	i++) {
				w=weight+woff(i,depth);	s=0;
				for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
				o[i]=(s*wh-y[i])*eta;
			}
			type	*g=d+(depth-1)*hidden;
			for(unsigned	i=0;	i<output;	i++) {
				w=weight+woff(i,depth);	gr=grd+woff(i,depth);	s=o[i];
				for(unsigned  j=0;  j<hidden; j++) {	g[j]+=s*w[j];	gr[j]-=s*p[j];	}
			}
		}
		for(unsigned	l=depth-1;	l;	l--) {
			type	*p=a+(l-1)*hidden,	*q=a+l*hidden,	*g=d+(l-1)*hidden,	*h=d+l*hidden;
			for(unsigned	i=0;	i<hidden;	i++) {
				w=weight+woff(i,l);	gr=grd+woff(i,l);	s=h[i]*gra(q[i])*wh;
				for(unsigned  j=0;  j<hidden; j++) {	g[j]+=s*w[j];	gr[j]-=s*p[j];	}
			}
		}
		gr=grd+woff(input,0);
		for(unsigned	i=0;	i<hidden;	i++){ d[i]*=gra(a[i])*wi;	gr[i]-=d[i];	}
		for(unsigned  i=0;  i<input; i++)	{
			gr=grd+woff(i,0);	s=x[i];
			for(unsigned	j=0;	j<hidden;	j++)	gr[j]-=s*d[j];
		}
		accu[tid]++;
		if((wy32x32(accu[tid],tid)&1023)==0){
			unsigned	si=size();
			omp_set_lock(&lock);
			for(unsigned	i=0;	i<si;	i++)	weight[i]+=grd[i];
			omp_unset_lock(&lock);
			memset(grd,0,size()*sizeof(type));
		}
	}
	void	predict(float	*x,	float	*y) {
		const	type	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
		type	a[depth*hidden]= {},	*w,	s;
		for(unsigned  i=0;  i<input; i++)	{
			w=weight+woff(i,0);	s=x[i];
			for(unsigned	j=0;	j<hidden;	j++)	a[j]+=s*w[j];
		}
		w=weight+woff(input,0);
		for(unsigned	i=0;	i<hidden;	i++) a[i]=act(wi*(a[i]+w[i]));
		a[0]=1;
		for(unsigned	l=1;	l<depth;	l++) {
			type	*p=a+(l-1)*hidden,	*q=a+l*hidden;
			for(unsigned	i=0;	i<hidden;	i++) {
				w=weight+woff(i,l);	s=0;
				for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
				q[i]=act(s*wh);
			}
			q[0]=1;
		}
		type	*p=a+(depth-1)*hidden;
		for(unsigned	i=0;	i<output;	i++) {
			w=weight+woff(i,depth);	s=0;
			for(unsigned	j=0;	j<hidden;	j++)	s+=w[j]*p[j];
			y[i]=s*wh;
		}
	}
};
