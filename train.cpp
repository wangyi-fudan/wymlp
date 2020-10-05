#include	<sys/time.h>
#include	<algorithm>
#include	<iostream>
#include	<stdint.h>
#include	<unistd.h>
#include	<fstream>
#include	"wymlp.hpp"
#include	<vector>
using	namespace	std;
const	unsigned	fullbatch=1<<20;
wymlp<32,2,1>	model;

bool	load_matrix(const	char	*F,	vector<float>	&M,	unsigned	&R,	unsigned	&C) {
	ifstream	fi(F);
	if(!fi) {	cerr<<"fail to open "<<F<<'\n';	return	false;	}
	string	buf;	R=C=0;
	while(getline(fi,buf))	if(buf.size()) {
		char	*p=(char*)buf.data(),	*q;
		for(;;) {	
			q=p;	float	x=strtod(p,	&p);
			if(p!=q)	M.push_back(x);	else	break;
		}
		R++;
	}
	fi.close();
	if(M.size()%R) {	cerr<<"unequal column\t"<<F<<'\n';	return	false;	}
	C=M.size()/R;	cerr<<F<<'\t'<<R<<'*'<<C<<'\n';
	return	true;
}

void	document(void) {
	cerr<<"Usage:	train [options] X Y\n";
	cerr<<"\t-e:	learning rate	default=0.01\n";
	cerr<<"\t-n:	#epoches	default=16\n";
	exit(0);
}

int	main(int	ac,	char	**av){
	uint64_t	seed=0;
	cerr<<"***********************************\n";
	cerr<<"* train                           *\n";
	cerr<<"* author: Yi Wang                 *\n";
	cerr<<"* email:  godspeed_china@yeah.net *\n";
	cerr<<"* date:   25/Oct/2019             *\n";
	cerr<<"***********************************\n";
	float	eta=0.01;	size_t	epoches=16;
	int	opt;
	while((opt=getopt(ac,	av,	"e:n:"))>=0) {
		switch(opt) {
		case	'e':	eta=atof(optarg);	break;
		case	'n':	epoches=atoi(optarg);	break;
		default:	document();
		}
	}
	if(ac<optind+2)	document();

	vector<float>	xmat,	ymat,	data;	unsigned	sample,	sample1,	xsize,	ysize,	feature;
	if(!load_matrix(av[optind],	xmat,	sample,	xsize))	return	0;
	if(!load_matrix(av[optind+1],	ymat,	sample1,	ysize))	return	0;
	if(sample!=sample1)	return	0;
	feature=ysize+xsize;
	data.resize(sample*feature);
	for(size_t	i=0;	i<sample;	i++){
		for(size_t	j=0;	j<ysize;	j++)	data[i*feature+j]=ymat[i*ysize+j];
		for(size_t	j=0;	j<xsize;	j++)	data[i*feature+ysize+j]=xmat[i*xsize+j];
	}
	vector<float>().swap(xmat);	vector<float>().swap(ymat);
	vector<float>	mean(feature),	prec(feature);
	for(size_t	j=0;	j<feature;	j++){
		double	sx=0,	sxx=0;
		for(size_t	i=0;	i<sample;	i++){	float	x=data[i*feature+j];	sx+=x;	sxx+=x*x;	}
		sx/=sample;	sxx=sqrt(sxx/sample-sx*sx);	float	m=mean[j]=sx,	p=prec[j]=sxx>0?1/sxx:0;
		for(size_t	i=0;	i<sample;	i++)	data[i*feature+j]=(data[i*feature+j]-m)*p;
	}
	vector<bool>	trte;	for(size_t	i=0;	i<sample;	i++)	trte.push_back(wy32x32(i,0)&7);
	model.input=xsize;	model.alloc_weight();	model.init_weight(seed);

	timeval	beg,	end;	gettimeofday(&beg,NULL);
	for(size_t	e=0;	e<epoches;	e++){
		for(size_t	i=0;	i<fullbatch;	i++){
			uint64_t	ran;
			do	ran=wyrand(&seed)%sample;	while(!trte[ran]);
			model.model(data.data()+ran*feature+ysize,	data.data()+ran*feature,	eta);
		}
		double	loss=0,	n=0;
		for(size_t	i=0;	i<sample;	i++)	if(!trte[i]){
			float	h=0,	t=data[i*feature];
			model.model(data.data()+i*feature+ysize,	&h,	-1);
			loss+=(h-t)*(h-t);	n+=1;
		}
		cerr<<e<<'\t'<<sqrt(loss/n)/prec[0]<<'\n';
	}
	gettimeofday(&end,NULL);
	float	deltat=(end.tv_sec-beg.tv_sec)+1e-6*(end.tv_usec-beg.tv_usec);
	cerr<<epoches*fullbatch/deltat<<" sample/sec\n";
	cerr<<epoches*fullbatch*1e-9*model.flops()/deltat<<" GFLOPs\n";
	model.free_weight();
	return	0;
}
