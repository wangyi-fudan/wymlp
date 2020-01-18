#include	"wypair.hpp"
#include	<iostream>
#include	<fstream>
#include	<vector>
using	namespace	std;

bool	load_matrix(const	char	*F,	vector<float>	&M,	uint64_t	&R,	uint64_t	&C) {
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

int	main(int	ac,	char	**av){
	if(ac<2){	cerr<<"train input [ output epoches(~1000) learning_rate(~0.01) cov]\n";	return	0;	}
	string	out=ac>2?av[2]:"wypca.txt";
	uint64_t	iters=ac>3?atoi(av[3]):1024;
	float	eta=ac>4?atof(av[4]):0.01;	
	bool	cov=ac>5?1:0;
	vector<float>	data;	uint64_t	sample,	feature;
	if(!load_matrix(av[1],	data,	sample,	feature))	return	0;
	for(size_t	j=0;	j<feature;	j++){
		double	sx=0,	sxx=0;
		for(size_t	i=0;	i<sample;	i++){	float	x=data[i*feature+j];	sx+=x;	sxx+=x*x;	}
		sx/=sample;	sxx=sqrt(sxx/sample-sx*sx);	if(sxx>0)	sxx=1/sxx;
		if(cov){
			for(size_t	i=0;	i<sample;	i++)	data[i*feature+j]=data[i*feature+j]-sx;
		}
		else{
			for(size_t	i=0;	i<sample;	i++)	data[i*feature+j]=(data[i*feature+j]-sx)*sxx;
		}
	}
	float	scalex=sqrt(sample),	scaley=sqrt(feature);
	vector<float>	w(wypair<wy_input,wy_hidden,wy_depth,0>(NULL,sample,feature,0,0,0,0));
	wysrand(0);	for(size_t	i=0;	i<w.size();	i++)	w[i]=wy2gau(wygrand());
	cerr.setf(ios::fixed);	cerr.precision(4);
	cerr<<"paras\t"<<(float)w.size()/(sample+feature)<<'\n';
	for(size_t	it=0;	it<iters;	it++){
		double	loss=0;
		for(size_t	i=0;	i<sample*feature;	i++){
			uint64_t	s=wygrand()%sample,	f=wygrand()%feature;
			loss+=wypair<wy_input,wy_hidden,wy_depth,0>(w.data(),s,f+sample,data[s*feature+f],eta,scalex,scaley);
		}
		cerr<<it<<'\t'<<"R2="<<1-loss/sample/feature<<" \r";
		eta*=1-2.0/iters;
	}
	float	*p=w.data()+w.size()-(sample+feature)*wy_input;
	ofstream	fo(out.c_str());
	for(size_t	i=0;	i<sample;	i++){
		for(size_t	j=0;	j<wy_input;	j++)	fo<<p[i*wy_input+j]<<'\t';
		fo<<'\n';
	}
	fo.close();
	cerr<<'\n';
	return	0;
}
