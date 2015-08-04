
def gen_datestr():
    from datetime import datetime   
    dt = datetime.now()
    return str(dt.year)+'.'+str(dt.month)+'.'+str(dt.day)+'_'+str(dt.hour)+'.'+str(dt.minute)+'.'+str(dt.second)