import numpy
import matplotlib.pyplot as plt
import sys
from numpy import trapz

def autocorr1(x,lags):
    '''numpy.corrcoef, partial'''

    corr=[1. if l==0 else numpy.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return numpy.array(corr)

def autocorr2(x,lags):
    '''manualy compute, non partial'''

    mean=numpy.mean(x)
    var=numpy.var(x)
    xp=x-mean
    corr=[1. if l==0 else numpy.sum(x[l:]*x[:-l]) for l in lags]

    return numpy.array(corr)

def autocorr3(x,lags):
    '''fft, pad 0s, non partial'''

    n=len(x)
    # pad 0s to 2n-1
    ext_size=2*n-1
    # nearest power of 2
    fsize=2**numpy.ceil(numpy.log2(ext_size)).astype('int')

    xp=x-numpy.mean(x)
    var=numpy.var(x)

    # do fft and ifft
    cf=numpy.fft.fft(xp,fsize)
    sf=cf.conjugate()*cf
    corr=numpy.fft.ifft(sf).real
    corr=corr/var/n

    return corr[:len(lags)]

def autocorr4(x,lags):
    '''fft, don't pad 0s, non partial'''
    mean=x.mean()
    var=numpy.var(x)
    xp=x-mean

    cf=numpy.fft.fft(xp)
    sf=cf.conjugate()*cf
    corr=numpy.fft.ifft(sf).real/var/len(x)

    return corr[:len(lags)]

def autocorr5(x,lags):
    '''numpy.correlate, non partial'''
    mean=x.mean()
    var=numpy.var(x)
    xp=x-mean
    corr=numpy.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)

    return corr[:len(lags)]
f=sys.argv[1]

if __name__=='__main__':
    y=numpy.loadtxt(f)	
    #y=[28,28,26,19,16,24,26,24,24,29,29,27,31,26,38,23,13,14,28,19,19,\
    #        17,22,2,4,5,7,8,14,14,23]
    y=numpy.array(y).astype('float')

    lags=range(1,31)
    fig,ax=plt.subplots()

    for funcii, labelii in zip([autocorr2], ['manual, non-partial']):

        cii=funcii(y,lags)
	area = trapz(cii)
        print("area =", area)
        ax.plot(lags,cii,label=labelii)
	print (area)
    ax.set_xlabel('lag')
    ax.set_ylabel('correlation coefficient')
    ax.legend()
    plt.show()
