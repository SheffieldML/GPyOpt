# Copyright (c) 2014, GPyOpt authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.interface import parser, BODriver, ObjectiveFunc

if __name__ == '__main__':
    import sys,os
    if len(sys.argv)<1:
        print 'Need the config file!'
        exit()
    
    configfile = sys.argv[1]
    curpath =  os.path.dirname(os.path.abspath(configfile))
    
    config = parser(configfile)
    config['prjpath'] = curpath
    obj_func = ObjectiveFunc(config)
    driver = BODriver(config, obj_func)
    driver.run()
