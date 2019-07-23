import GPy
import GPyOpt
import time 
import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Process, Queue, Lock
from GPyOpt.experiment_design import initial_design
from GPyOpt.acquisitions.EI import AcquisitionEI
from GPyOpt.methods import ModularBayesianOptimization

def rosenbrock(x,y):
  return ((1.0-x)**2+100*((y-x**2)**2))

def line_constraint(x,y):
  return (2.-x-y)

def cube_constraint(x,y):
  return (y+(1.-x)**3 - 1.)

def consumer(query_queue,result_queue, lock):
    #with lock:
    #    print('Starting consumer {}'.format( mp.current_process() ))
    
    while True:
        # If the queue is empty, queue.get() will block until the queue has data
        point = query_queue.get()

        #with lock:
        #  print('Consumer {} received point {} to process'.format(mp.current_process(),point))
        
        #time.sleep(random.random() * 5)
        
        if point is None:
            with lock: 
                print('Consumer {} exiting ...'.format( mp.current_process() ))
            
            return
        
        y  = rosenbrock(point[0],point[1])
        c1 = line_constraint(point[0],point[1])
        c2 = cube_constraint(point[0],point[1])
        
        result = np.hstack((point,np.array([y,c1,c2]))) 
                
        #with lock:
        #    print('Consumer {} got {} point with result {}'.format(mp.current_process(),point,result))
        
        result_queue.put(result)

def init_producer(query_queue,lock,num_init,X_init):
    with lock:
        print('Starting init_producer {}'.format( mp.current_process() ))
    
    for i in range(num_init):
        point = X_init[i,:]
        query_queue.put(point)
    
    #with lock:
    #    print('init_producer {} exiting'.format( mp.current_process() ))
        
    
def BO_loop_producer(query_queue, result_queue, lock,
                     num_init, iter_count,num_threads, 
                     model, model_c, space, objective,
                     constraint, acquisition, return_dict):
  
  np.set_printoptions(precision=2)

  with lock:
    print('Starting BO_loop_producer {}'.format( mp.current_process() ))

  X_init = np.zeros((num_init,2))
  Y_init = np.zeros((num_init,1))
  C_init = np.zeros((num_init,2))

  current_iter = 0
  for i in range(num_init):
    result = result_queue.get()
    X_init[i,0:2] = result[0:2]
    Y_init[i,0:1] = result[2:3]
    C_init[i,0:2] = result[3:5]
    
  X_step = X_init
  Y_step = Y_init
  C_step = C_init 

  # Initializer loop, to atribute the first evaluation points 
  ThompsonBatchEvaluator = GPyOpt.core.evaluators.ThompsonBatch(acquisition,batch_size = num_threads)
  bo_init = ModularBayesianOptimization(model, space, objective, acquisition, ThompsonBatchEvaluator, 
                                        X_init = X_step, Y_init = Y_step, C_init = C_step, 
                                        model_c = model_c, normalize_Y = False)
    
  x_next = bo_init.suggest_next_locations()
  for i in range(num_threads):
    point = x_next[i,:]
    query_queue.put(point)
  current_iter += num_threads
    
  # Main loop, it must be run after initializer loop
  # --- CHOOSE a collection method
  SequentialEvaluator = GPyOpt.core.evaluators.Sequential(acquisition)
  while current_iter < iter_count:
    result = result_queue.get()
        
    x_eval = result[:X_step.shape[1]]
    y_eval = result[(X_step.shape[1]+0):(X_step.shape[1]+1)]
    c_eval = result[(X_step.shape[1]+1):]
        
    with lock:
      print("Eval - ",current_iter," : ",np.array([y_eval[0],c_eval[0],c_eval[1],x_eval[0],x_eval[1]]))
        
    X_step = np.vstack((X_step, x_eval))
    Y_step = np.vstack((Y_step, y_eval))
    C_step = np.vstack((C_step, c_eval))
        
    bo_step = ModularBayesianOptimization(model, space, objective, acquisition, SequentialEvaluator, 
                                          X_init = X_step, Y_init = Y_step, C_init = C_step, 
                                          model_c = model_c, normalize_Y = False)
    x_next = bo_step.suggest_next_locations()
    
    query_queue.put(x_next[0])

    current_iter += 1
    
  with lock:
    print('BO_loop_producer {} exiting'.format( mp.current_process() ))
  
  return_dict[0] = np.hstack((X_step,Y_step,C_step))

if __name__ == '__main__':
  np.set_printoptions(precision=2)

  objective = None
  constraint = None
  
  space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-1.5,1.5)},
          {'name': 'var_2', 'type': 'continuous', 'domain': (-0.5,2.5)}]
  space = GPyOpt.Design_space(space = space)
  
  kernel   = None 
  kernel_c = None
  
  model    = GPyOpt.models.GPModel(optimize_restarts=5,verbose=False)
  model_c1 = GPyOpt.models.GPModel(optimize_restarts=5,verbose=False)
  model_c2 = GPyOpt.models.GPModel(optimize_restarts=5,verbose=False)
  model_c  = [model_c1,model_c2]
  
  aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
  
  acquisition = AcquisitionEI(model,space,optimizer=aquisition_optimizer,jitter = 1e-3,
                              model_c=model_c,jitter_c = np.array([0.0,0.0]))
                                  
  num_init = 23
  iter_count = 150-num_init
  typ_init = 'latin'
  X_init = initial_design(typ_init,space,num_init)
  dataset = None

  num_threads = 12
  lock  = Lock()
  query_queue  = Queue()
  result_queue = Queue()
  
  consumers = []
    
  initializer = Process(target=init_producer,args=(query_queue,lock,num_init,X_init))
  
  initializer.start()
 
  for i in range(num_threads):
    p = Process(target=consumer,args=(query_queue,result_queue,lock))
    consumers.append(p)

  for c in consumers:
    c.start()

  initializer.join()

  manager = mp.Manager()
  return_dict = manager.dict()

  BO_args = (query_queue, result_queue, lock, num_init, iter_count,num_threads, model, model_c, space, objective, constraint, acquisition, return_dict)
  BO_loop = Process(target=BO_loop_producer,args=BO_args)
  BO_loop.start()

  BO_loop.join()

  print(return_dict[0])

  for i in range(num_threads):
    query_queue.put(None)

  for c in consumers:
    c.join()


