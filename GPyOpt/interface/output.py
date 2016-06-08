# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

class DataSaver(object):
    
    def __init__(self, config, outpath=None, prjname='', name=''):
        interval = config['interval']
        assert interval>0 or interval==-1
        self.interval=interval
        self.outpath = outpath
        self.prjname = prjname
        self.name = name

    def save_data(self, iters, times, offsets, X, Y, bo):
        pass
    
    def close(self):
        pass

class Report(DataSaver):
    def __init__(self, config, outpath, prjname='',name=''):
        super(Report, self).__init__(config, outpath, prjname,name)
        import os
        from ..util.io import gen_datestr
        filename =self.name+'_'+self.prjname+'_' +gen_datestr()+'.txt' if config['filename'] is None else config['filename']
        self.filename = os.path.join(self.outpath,filename)
            
    def save_data(self, iters, times, offsets, X, Y, bo):
        import time
        
        with open(self.filename,'w') as file:
            file.write('---------------------------------' + ' Results file ' + '--------------------------------------\n')
            file.write('GPyOpt Version 0.1.0 \n')
            file.write('Date and time:              ' + time.strftime("%c")+'\n')
            file.write('Optimization completed:     '+ str(bo.X.shape[0]).strip('[]') + ' samples collected.\n')
            file.write('Optimization time:          ' + str(bo.time).strip('[]') +' seconds.\n') 
    
            file.write('---------------------------------' + ' Problem set up ' + '------------------------------------\n')
            file.write('Problem Dimension:          ' + str(bo.input_dim).strip('[]') +'\n')    
            file.write('Problem bounds:             ' + str(bo.bounds).strip('[]') +'\n') 
            file.write('Batch size:                 ' + str(bo.n_inbatch).strip('[]') +'\n')    
            file.write('Acquisition:                ' + bo.acqu_name + '\n')  
            file.write('Acquisition optimizer:      ' + bo.acqu_optimize_method+ '\n')  
            file.write('Sparse GP:                  ' + str(bo.sparseGP).strip('[]') + '\n')  
            file.write('---------------------------------' + ' Summary ' + '------------------------------------------\n')
            file.write('Best found minimum:         ' + str(min(bo.Y)).strip('[]') +'\n') 
            file.write('Minumum location:           ' + str(bo.X[np.argmin(bo.Y),:]).strip('[]') +'\n') 
    
            file.close()
            
class Logger(DataSaver):
    def __init__(self, config, outpath, prjname='',name=''):
        super(Logger, self).__init__(config, outpath, prjname,name)
        import os
        from ..util.io import gen_datestr
        assert config['format'].lower()=='csv', 'Data logger '+self.name+': unsupported format '+config['format']+'!'
        assert config['content'].lower()=='ybest', 'Data logger '+self.name+': unsupported content '+config['content']+'!'
        filename =self.name+'_'+self.prjname+'_' +gen_datestr()+'.csv' if config['filename'] is None else config['filename']
        self.filename = os.path.join(self.outpath,filename)
        self.fileout = None
        self.write_headline = True
        try:
            self.fileout = open(self.filename,'w')
        except:
            print('Data logger '+self.name+' fails to open the output file '+self.filename+'!')
    
    def close(self):
        if self.fileout is not None:
            self.fileout.close()
        
    def save_data(self, iters, times, offsets, X, Y, bo):
        if self.fileout is None: return
        if self.write_headline:
            items = ['iteration', 'time(sec)', 'objective']+['input['+str(i+1)+']' for i in range(X[0].shape[1])]
            self.fileout.write(','.join(items)+'\n')
            self.write_headline = False
        Y = np.vstack(Y)
        X = np.vstack(X)
        idx =  np.argmin(Y)
        items = [str(iters[-1]), str(times[-1]), str(float(Y[idx]))]+[str(X[idx,i]) for i in range(X.shape[1])]
        self.fileout.write(','.join(items)+'\n')
        
class OutputEng(object):
    
    _support_savers = {
                            'report': Report,
                            'logger': Logger,
                            }
    
    def __init__(self, config):
        self. config = config
        
        self.times = []
        self.iters = []
        self.offsets = []
        self.Xs = []
        self.Ys = []
        
        # create all the data savers
        self.data_savers = [self._support_savers[ds['type']](ds, config['prjpath'], config['experiment-name'],name) for name, ds in config['output'].iteritems() if isinstance(ds, dict)]
        
        self.clock = [ds.interval for ds in self.data_savers]
        
    def append_iter(self, iters, elapsed_time, X, Y, bo, final=False):
        self.iters.append(iters)
        self.times.append(elapsed_time)
        self.Xs.append(X)
        self.Ys.append(Y)
        self.offsets.append(X.shape[0])
        
        for i in xrange(len(self.data_savers)):
            if final:
                if self.clock[i]==-1: self.clock[i] = 0
            elif self.clock[i]>0: self.clock[i] += -1
            if self.clock[i]==0:
                self.data_savers[i].save_data(self.iters, self.times, self.offsets, self.Xs, self.Ys, bo)
                self.clock[i] = self.data_savers[i].interval
                
    def close(self):
        for ds in self.data_savers: ds.close()
        

