

class OutputEng(object):
    
    def __init__(self, config):
        self. config = config
        
        self.times = []
        self.iters = []
        self.Xs = []
        self.Ys = []
        
    def append_iter(self, iters, elapsed_time, X, Y):
        self.iters.append(iters)
        self.times.append(elapsed_time)
        self.Xs.append(X)
        self.Ys.append(Y)
        
        