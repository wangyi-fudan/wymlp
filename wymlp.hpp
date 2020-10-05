#include	<sys/mman.h>
#include	<sys/stat.h>
#include	"wyhash32.h"
#include	<string.h>
#include	<unistd.h>
#include	<float.h>
#include	<stdio.h>
#include	<fcntl.h>
#include	<math.h>
template<unsigned	hidden,	unsigned	depth,	unsigned	output,	unsigned	type>
class	wymlp {
private:
	int	fd;
	struct	stat	sb;
	float	act(float	x) {	return  x/(1+fabsf(x));	}
	float	gra(float	x) {	x=1-fabsf(x);	return	x*x;	}
	unsigned	woff(unsigned	i,	unsigned	l) {	return	 (input+1)*hidden+(l-1)*hidden*hidden+i*hidden;	}
public:
	unsigned	input;
	float	*weight;
	wymlp() {	weight=NULL;	}
	unsigned	flops(void) {	return	 (input+1)*hidden*4+((depth-1)*hidden*hidden+output*hidden)*6;	}
	unsigned	size(void) {	return	(input+1)*hidden+(depth-1)*hidden*hidden+output*hidden;	}
	void	alloc_weight(void) {	free(weight);	weight=(float*)aligned_alloc(64,size()*sizeof(float));	}
	void	free_weight(void) {	free(weight);	weight=NULL;	}
	void	init_weight(uint64_t	seed) {	for(unsigned	i=0;	i<size();	i++)	weight[i]=wy2gau(wyrand(&seed));	}
	bool	mmap_weight(const	char	*F) {
		fd=open(F,	O_RDONLY);
		if(fd<0)	return	false;
		fstat(fd,	&sb);
		weight=(float*)mmap(NULL,	sb.st_size,	PROT_READ,	MAP_SHARED,	fd,	0);
		if(weight==MAP_FAILED)	return	false;
		return	true;
	}
	void	munmap_weight(void) {
		munmap(weight,sb.st_size);
		close(fd);
		weight=NULL;
	}
	bool	save(const	char	*F) {
		FILE	*f=fopen(F,	"wb");
		if(f==NULL)	return	false;
		fwrite(weight,sizeof(float),size(),f);
		fclose(f);
		return	true;
	}
	bool	load(const	char	*F) {
		if(weight==NULL)	alloc_weight();
		FILE	*f=fopen(F,	"rb");
		if(f==NULL)	return	false;
		fread(weight,sizeof(float),size(),f);
		fclose(f);
		return	true;
	}
	void	model(float	*x,	float	*y,	float	eta,	uint64_t	seed,	double	drop) {
		const	float	wh=1/sqrtf(hidden),	wi=1/sqrtf(input+1);
		float	a[2*depth*hidden+output]= {},	*d=a+depth*hidden,	*o=d+depth*hidden,	*w,	s,	*p,	*q,	*g,	*h,	m;
		unsigned	i,j,l;
		uint64_t	s0=seed;
		for(i=0;  i<=input; i++) {
			w=weight+i*hidden;
			if(eta>0)	s=i<input?(wy2u01(wyrand(&seed))<drop?0:x[i]):1;
			else	s=i<input?((1-drop)*x[i]):1;
			if(s)	for(j=0;	j<hidden;	j++)	a[j]+=s*w[j];
		}
		for(i=0;	i<hidden;	i++) {	a[i]=act(wi*a[i]);	}	a[0]=1;
		for(l=1;	l<=depth;	l++) {
			p=a+(l-1)*hidden;	q=l<depth?a+l*hidden:o;	q[0]=1;
			for(i=l<depth?1:0;	i<(l<depth?hidden:output);	i++) {
				w=weight+woff(i,l);	s=0;
				for(j=0;	j<hidden;	j++)	s+=w[j]*p[j];
				q[i]=l<depth?act(wh*s):wh*s;
			}
		}
		switch(type) {
		case	0:
			break;
		case	1:
			for(i=0;    i<output;   i++)	o[i]=1/(1+expf(-o[i]));
			break;
		case	2: {
			m=-FLT_MAX;	s=0;
			for(i=0;    i<output;   i++)	if(o[i]>m)	m=o[i];
			for(i=0;    i<output;   i++)	s+=(o[i]=expf(o[i]-m));
			for(i=0;    i<output;   i++)	o[i]/=s;
		}
		break;
		}
		if(eta<0) {
			for(i=0;    i<output;   i++)	y[i]=o[i];
			return;
		}
		switch(type) {
		case	0:
		case	1:
			for(i=0;    i<output;   i++)	o[i]=(o[i]-y[i])*eta;
			break;
		case	2:
			for(i=0;    i<output;   i++)	o[i]=(o[i]-(i==y[0]))*eta;
			break;
		}
		for(l=depth;	l;	l--) {
			p=a+(l-1)*hidden;	q=l<depth?a+l*hidden:o;	g=d+(l-1)*hidden;	h=l<depth?d+l*hidden:o;
			for(i=l<depth?1:0;	i<(l<depth?hidden:output);	i++) {
				w=weight+woff(i,l);	s=(l<depth?gra(q[i]):1)*h[i]*wh;
				for(j=0;  j<hidden; j++) {	g[j]+=s*w[j];	w[j]-=s*p[j];	}
			}
		}
		for(i=0;	i<hidden;	i++) d[i]*=gra(a[i])*wi;
		for(i=0;  i<=input; i++) {
			w=weight+i*hidden;	s=i<input?(wy2u01(wyrand(&s0))<drop?0:x[i]):1;
			if(s)	for(j=0;	j<hidden;	j++)	w[j]-=s*d[j];
		}
	}
};
