# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

class DataSaver(object):
    
    def __init__(self, config, outpath=None, prjname=''):
        interval = config['interval']
        assert interval>0 or interval==-1
        self.interval=interval
        self.outpath = outpath
        self.prjname = prjname

    def save_data(self, iters, times, offsets, X, Y, bo):
        pass

class Report(DataSaver):
    def __init__(self, config, outpath, prjname=''):
        super(Report, self).__init__(config, outpath, prjname)
        import os
        from ..util.io import gen_filename_withdate
        filename = gen_filename_withdate(self.prjname) if config['filename'] is None else config['filename']
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
            
class OutputEng(object):
    
    _support_savers = {
                            'report': Report,
                            }
    
    def __init__(self, config):
        self. config = config
        
        self.times = []
        self.iters = []
        self.offsets = []
        self.Xs = []
        self.Ys = []
        
        # create all the data savers
        self.data_savers = [self._support_savers[ds['type']](ds, config['prjpath'], config['experiment-name']) for name, ds in config['output'].iteritems() if isinstance(ds, dict)]
        
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
        

