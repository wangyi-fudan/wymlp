//data download:	http://yann.lecun.com/exdb/mnist/
#include	<sys/time.h>
#include	<iostream>
#include	"wymlp.hpp"
#include	<vector>
#include	<zlib.h>
using	namespace	std;
const	unsigned	feature=784;
wymlp<128,2,10,2>	model;	

bool	load_image(const	char	*F,	vector<float>	&D,	unsigned	N){
	gzFile	in=gzopen(F,	"rb");
	if(in==Z_NULL)	return	false;
	unsigned	n;	gzread(in,	&n,	4);	gzread(in,	&n,	4);	gzread(in,	&n,	4);	gzread(in,	&n,	4);
	D.resize(N*feature);	vector<uint8_t>	temp(feature);
	for(size_t	i=0;	i<N;	i++){
		gzread(in,	temp.data(),	feature);
		for(size_t	j=0;	j<feature;	j++)	D[i*feature+j]=temp[j];
	}
	gzclose(in);	cerr<<F<<'\n';	return	true;
}

bool	load_label(const	char	*F,	vector<float>	&D,	unsigned	N){
	gzFile	in=gzopen(F,	"rb");
	if(in==Z_NULL)	return	false;
	unsigned	n;	gzread(in,	&n,	4);	gzread(in,	&n,	4);	D.resize(N);	uint8_t	temp;
	for(size_t	i=0;	i<N;	i++){	gzread(in,	&temp,	1);	D[i]=temp;	}
	gzclose(in);	cerr<<F<<'\n';	return	true;
}

int	main(int	ac,	char	**av){
	cerr<<"***********************************\n";
	cerr<<"* MNIST                           *\n";
	cerr<<"* author: Yi Wang                 *\n";
	cerr<<"* email:  godspeed_china@yeah.net *\n";
	cerr<<"* date:   29/Oct/2019             *\n";
	cerr<<"***********************************\n";
	float	eta=1;
	vector<float>	trainx,	trainy,	testx,	testy;	unsigned	trainn=60000,	testn=10000;	
	if(!load_image("train-images-idx3-ubyte.gz",	trainx,	trainn))	return	0;
	if(!load_image("t10k-images-idx3-ubyte.gz",	testx,	testn))	return	0;
	if(!load_label("train-labels-idx1-ubyte.gz",	trainy,	trainn))	return	0;
	if(!load_label("t10k-labels-idx1-ubyte.gz",	testy,	testn))	return	0;
	double	sx=0,	sxx=0,	sn=(trainn+testn)*feature;
	for(size_t	i=0;	i<trainx.size();	i++){	sx+=trainx[i];	sxx+=trainx[i]*trainx[i];	}
	for(size_t	i=0;	i<testx.size();	i++){	sx+=testx[i];	sxx+=testx[i]*testx[i];	}
	sx/=sn;	sxx=1/sqrt(sxx/sn-sx*sx);
	for(size_t	i=0;	i<trainx.size();	i++)	trainx[i]=sxx*(trainx[i]-sx);
	for(size_t	i=0;	i<testx.size();	i++)	testx[i]=sxx*(testx[i]-sx);
	uint64_t	seed=wy32x32(time(NULL),0);
	model.input=feature;	model.alloc_weight();	model.init_weight(seed);
	double	t0=0;
	for(size_t	it=0;	eta>0.001;	it++,eta*=0.97){
		timeval	beg,	end;	gettimeofday(&beg,NULL);
		for(size_t	i=0;	i<trainn;	i++){
			size_t	ran=wyrand(&seed)%trainn;
			model.model(trainx.data()+ran*feature,	trainy.data()+ran,	eta,	wyrand(&seed),0.5);
		}
		gettimeofday(&end,NULL);
		size_t	err=0;	float	p[10];
		for(size_t	i=0;	i<testn;	i++){
			model.model(testx.data()+i*feature,	p,	-1,	wyrand(&seed),0.5);
			uint8_t	pre=0;
			for(size_t	j=0;	j<10;	j++)	if(p[j]>p[pre])	pre=j;
			err+=pre!=testy[i];
		}
		cerr.precision(3);	cerr.setf(ios::fixed);	t0+=(end.tv_sec-beg.tv_sec)+1e-6*(end.tv_usec-beg.tv_usec);
		cerr<<it<<'\t'<<"error="<<100.0*err/testn<<"%\teta="<<eta<<"\ttime="<<t0<<"s\n";
	}
	model.free_weight();
	return	0;
}
