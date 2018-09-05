# cic3 class 
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 03.09.2018 19:26
import os
import sys
import numpy as np
import scipy.signal as sig
import tempfile
import subprocess
import shlex
import time
#Add TheSDK to path. Importing it first adds the rest of the modules
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))
from functools import reduce
from thesdk import *

from refptr import *
from verilog import *

#Simple buffer template
class cic3(verilog,thesdk):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.proplist = [' '];    #properties that can be propagated from parent
        self.Rs_high = 160e6*8;          # sampling frequency
        self.Rs_low  = 4*20e6;          # sampling frequency
        self.integscale = 1023
        self.cic3shift = 0
        self.iptr_A = refptr();
        self.model='py';             #can be set externally, but is not propagated
        self._Z = refptr();
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;
        self.init()

    def init(self):
        ratio=int(self.Rs_high/self.Rs_low)
        #Pervert reduce to convolve three FIRs 
        self.H=reduce(lambda val,cum: np.convolve(val[:,0],cum[:,0]).reshape(-1,1),[np.ones((ratio,1))]*3)
        self.def_verilog()
        self._vlogparameters=dict([ ('g_rs',self.Rs_high), ('g_Rs_slow',self.Rs_low), ('g_integscale',self.integscale) ])

    def main(self):
        ratio=int(self.Rs_high/self.Rs_low)
        #Convert this to cumsum, shift and diff
        out=np.convolve(self.iptr_A.Value.reshape(-1,1)[:,0],self.H[:,0]).reshape(-1,1)[0::ratio,0].reshape(-1,1)
        out=out*2**self.cic3shift
        if self.par:
            queue.put(out)
        self._Z.Value=out


    def run(self,*arg):
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            queue=arg[0]  #multiprocessing.Queue as the first argument
        else:
            self.par=False

        if self.model=='py':
            self.main()
        else: 
          self.write_infile()
          self.run_verilog()
          self.read_outfile()

    def write_infile(self):
        rndpart=os.path.basename(tempfile.mkstemp()[1])
        if self.model=='sv':
            self._infile=self._vlogsimpath +'/A_' + rndpart +'.txt'
            self._outfile=self._vlogsimpath +'/Z_' + rndpart +'.txt'
        elif self.model=='vhdl':
            pass
        else:
            pass
        try:
          os.remove(self._infile)
        except:
          pass
        fid=open(self._infile,'wb')
        np.savetxt(fid,self.iptr_A.Value.reshape(-1,1).view(float),fmt='%i', delimiter='\t')
        fid.close()

    def read_outfile(self):
        fid=open(self._outfile,'r')
        out = np.loadtxt(fid,dtype=complex)
        #Of course it does not work symmetrically with savetxt
        out=(out[:,0]+1j*out[:,1]).reshape(-1,1) 
        fid.close()
        if self.par:
          queue.put(out)
        self._Z.Value=out
        os.remove(self._outfile)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from  cic3 import *
    from  f2_signal_gen import *
    from  f2_system import *
    t=thesdk()
    t.print_log({'type':'I', 'msg': "This is a testing template. Enjoy"})
    fsorig=20e6
    highrate=fsorig*16*2
    lowrate=fsorig*8
    integscale=256
    siggen=f2_signal_gen()
    fsindexes=range(int(lowrate/fsorig),int(highrate/fsorig),int(lowrate/fsorig))
    print(list(fsindexes))
    freqlist=[1.0e6, 0.45*fsorig]
    _=[freqlist.extend([i*fsorig-0.5*fsorig, i*fsorig+0.5*fsorig]) for i in list(fsindexes) ] 
    #freqlist=list(filter(lambda x: x < highrate/2, freqlist))
    print(freqlist)
    siggen.bbsigdict={ 'mode':'sinusoid', 'freqs':freqlist, 'length':2**14, 'BBRs':highrate };
    siggen.Users=1
    siggen.Txantennas=1
    siggen.init()
    #Mimic ADC
    bits=10
    insig=siggen._Z.Value[0,:,0].reshape(-1,1)
    insig=np.round(insig/np.amax([np.abs(np.real(insig)), np.abs(np.imag(insig))])*(2**(bits-1)-1))
    str="Input signal range is %i" %((2**(bits-1)-1))
    t.print_log({'type':'I', 'msg': str})
    h=cic3()
    h.Rs_high=highrate
    h.Rs_low=lowrate
    h.integscale=integscale 
    h.iptr_A.Value=insig
    h.model='sv'
    h.init()
    impulse=np.r_['0', h.H, np.zeros((1024-h.H.shape[0],1))]
    #h.export_scala() 
    h.run() 

    w=np.arange(1024)/1024*highrate
    spe1=np.fft.fft(impulse,axis=0)
    f=plt.figure(1)
    plt.plot(w,20*np.log10(np.abs(spe1)/np.amax(np.abs(spe1))))
    plt.ylim((-80,3))
    plt.grid(True)
    f.show()
    
    #spe3=np.fft.fft(h._Z.Value,axis=0)
    maximum=np.amax([np.abs(np.real(h._Z.Value)), np.abs(np.imag(h._Z.Value))])
    str="Output signal range is %i" %(maximum)
    t.print_log({'type':'I', 'msg': str})
    fs, spe3=sig.welch(h._Z.Value,fs=lowrate,nperseg=1024,return_onesided=False,scaling='spectrum',axis=0)
    print(fs)
    w=np.arange(spe3.shape[0])/spe3.shape[0]*lowrate
    ff=plt.figure(3)
    plt.plot(w,10*np.log10(np.abs(spe3)/np.amax(np.abs(spe3))))
    plt.ylim((-80,3))
    plt.grid(True)
    ff.show()

    #Required to keep the figures open
    input()
