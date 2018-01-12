# cic3 class 
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 11.01.2018 17:18
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
from rtl import *

#Simple buffer template
class cic3(rtl,thesdk):
    def __init__(self,*arg): 
        self.proplist = [ 'Rs' ];    #properties that can be propagated from parent
        self.Rs = 160e6*8;          # sampling frequency
        self.cic3Rs_slow = 20e6;          # sampling frequency
        self.integscale = 1023
        self.iptr_A = refptr();
        self.model='py';             #can be set externally, but is not propagated
        self._Z = refptr();
        self._classfile=os.path.dirname(os.path.realpath(__file__)) + "/"+__name__
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;
        self.init()

    def init(self):
        ratio=int(self.Rs/self.cic3Rs_slow)
        #Pervert reduce to convolve three FIRs 
        self.H=reduce(lambda val,cum: np.convolve(val[:,0],cum[:,0]).reshape(-1,1),[np.ones((ratio,1))]*3)
        self.H=self.H/np.abs(np.amax(self.H))
        self.def_rtl()
        rndpart=os.path.basename(tempfile.mkstemp()[1])
        self._infile=self._rtlsimpath +'/A_' + rndpart +'.txt'
        self._outfile=self._rtlsimpath +'/Z_' + rndpart +'.txt'
        self._rtlparameters=dict([ ('g_rs',self.Rs), ('g_Rs_slow',self.cic3Rs_slow), ('g_integscale',self.integscale) ])
        #self._rtlparameters=dict([('g_integscale',self.integscale) ])
        self._rtlcmd=self.get_rtlcmd()

    def decimate_input(self):
        ratio=int(self.Rs/self.cic3Rs_slow)
        out=np.convolve(self.iptr_A.Value.reshape(-1,1)[:,0],self.H[:,0]).reshape(-1,1)[0::ratio,0]
        if self.par:
            queue.put(out)
        self._Z.Value=out

    def get_rtlcmd(self):
        #the could be gathered to rtl class in some way but they are now here for clarity
        submission = ' bsub '  
        rtllibcmd =  'vlib ' +  self._workpath + ' && sleep 2'
        rtllibmapcmd = 'vmap work ' + self._workpath

        if (self.model is 'vhdl'):
            pass
        #    rtlcompcmd = ( 'vcom ' + self._rtlsrcpath + '/' + self._name + '.vhd '
        #                  + self._rtlsrcpath + '/tb_'+ self._name+ '.vhd' )
        #    rtlsimcmd =  ( 'vsim -64 -batch -t 1ps -g g_infile=' + 
        #                   self._infile + ' -g g_outfile=' + self._outfile + ' -g g_gain=' + str(self.integscale)  
        #                   + ' work.tb_' + self._name + ' -do "run -all; quit -f;"')
        #    rtlcmd =  submission + rtllibcmd  +  ' && ' + rtllibmapcmd + ' && ' + rtlcompcmd +  ' && ' + rtlsimcmd

        elif (self.model is 'sv'):
            rtlcompcmd = ( 'vlog -work work ' + self._rtlsrcpath + '/' + self._name + '.sv '
                           + self._rtlsrcpath + '/tb_' + self._name +'.sv')

            gstring=' '.join([ ('-g ' + str(param) +'='+ str(val)) for param,val in iter(self._rtlparameters.items()) ])
            rtlsimcmd = ( 'vsim -64 -batch -t 1ps -voptargs=+acc -g g_infile=' + self._infile
                          + ' -g g_outfile=' + self._outfile + ' ' + gstring
                          + ' work.tb_' + self._name  + ' -do "run -all; quit;"' )

            rtlcmd =  submission + rtllibcmd  +  ' && ' + rtllibmapcmd + ' && ' + rtlcompcmd +  ' && ' + rtlsimcmd

        else:
            rtlcmd=[]
        return rtlcmd

    def run(self,*arg):
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            queue=arg[0]  #multiprocessing.Queue as the first argument
        else:
            self.par=False

        if self.model=='py':
            self.decimate_input()
        else: 
          try:
              os.remove(self._infile)
          except:
              pass
          fid=open(self._infile,'wb')
          #np.savetxt(fid,np.transpose(self.iptr_A.Value),fmt='%.0f')
          np.savetxt(fid,self.iptr_A.Value.reshape(-1,1).view(float),fmt='%i', delimiter='\t')
          fid.close()
          while not os.path.isfile(self._infile):
              self.print_log({'type':'I', 'msg':"Wait infile to appear"})
              time.sleep(5)
          try:
              os.remove(self._outfile)
          except:
              pass
          self.print_log({'type':'I', 'msg':"Running external command %s\n" %(self._rtlcmd) })
          subprocess.call(shlex.split(self._rtlcmd));
          
          while not os.path.isfile(self._outfile):
              self.print_log({'type':'I', 'msg':"Wait outfile to appear"})
              time.sleep(5)
          fid=open(self._outfile,'r')
          out = np.loadtxt(fid,dtype=complex)
          #Of course it does not work symmetrically with savetxt
          out=(out[:,0]+1j*out[:,1]).reshape(-1,1) 
          fid.close()
          if self.par:
              queue.put(out)
          self._Z.Value=out
          os.remove(self._infile)
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
    integscale=4
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
    h.Rs=highrate
    h.cic3Rs_slow=lowrate
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
