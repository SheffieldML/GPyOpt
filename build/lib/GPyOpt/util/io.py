# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

def gen_datestr():
    """
    Returns a string with the yy/mm/dd and  hh/mm/ss
    """
    from datetime import datetime   
    dt = datetime.now()
    return str(dt.year)+'.'+str(dt.month)+'.'+str(dt.day)+'_'+str(dt.hour)+'.'+str(dt.minute)+'.'+str(dt.second)