import unittest
import numpy as np
import GPyOpt
from numpy.random import seed

class TestAcquisitions(unittest.TestCase):
    '''
    Unit test of the availabble acquisition functions and their gradients: EI, UCB, MPI.

    '''

    def setUp(self):
        #initialize model
        seed(1234)
        self.bjective  = GPyOpt.fmodels.experiments1d.forrester()
        bounds = [(0,1)]  

        self.bo = GPyOpt.methods.BayesianOptimization(f=self.bjective.f,   # function to optimize       
                                                    bounds=bounds,          # box-constrains of the problem
                                                    acquisition_type='EI')  
        self.bo.run_optimization(max_iter=1, eps=10e-6) 
        
    def test_bo(self):
        # Run bo optimization
        self.assertEqual(self.bo.suggested_sample[0][0],0.66656362406227299)






if __name__ == '__main__':
    unittest.main()
        