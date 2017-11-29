class ExperimentDesign(object):
    """
    Base class for all experiment designs
    """
    def __init__(self, space):
        self.space = space

    def get_samples(self, init_points_count):
        raise NotImplementedError("Subclasses should implement this method.")
