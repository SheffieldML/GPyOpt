#!/usr/bin/env python
# Copyright (c) 2014, GPyOpt authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.interface import parser, BODriver, load_objective, OutputEng

if __name__ == '__main__':
    import sys,os
    if len(sys.argv)<1:
        print('Need the config file!')
        exit()
    
    configfile = sys.argv[1]
    curpath =  os.path.dirname(os.path.abspath(configfile))
    
    config = parser(configfile)
    config['prjpath'] = curpath
    obj_func = load_objective(config)
    driver = BODriver(config, obj_func)
    bo = driver.run()
#     bo.save_report(os.path.join(curpath,config['experiment-name']+'_report.txt'))
    bo.save_evaluations(os.path.join(curpath,config['experiment-name']+'_evaluations.txt'))
    bo.save_models(os.path.join(curpath,config['experiment-name']+'_model.txt'))