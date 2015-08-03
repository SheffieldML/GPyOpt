
def gen_filename_withdate(prjname):
    from datetime import datetime   
    dt = datetime.now()
    return 'report_'+prjname+'_'+str(dt.year)+'.'+str(dt.month)+'.'+str(dt.day)+'_'+str(dt.hour)+'.'+str(dt.minute)+'.'+str(dt.second)+'.txt'