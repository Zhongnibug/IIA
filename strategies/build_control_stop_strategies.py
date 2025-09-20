import omegaconf

class ValNoBestDelayStop:
    def __init__(self, control_eval_class, delay_epochs, negative_direction=-1):
        self.control_eval_class = control_eval_class
        self.delay_epochs = delay_epochs
        self.negative_direction = negative_direction
        self.best_score = 10000.0 * self.negative_direction

        # Init control class
        if isinstance(self.control_eval_class, str):
            self.control_class = self.control_eval_class
            self.control_subclass = None
        elif isinstance(self.control_eval_class, list) or \
            isinstance(self.control_eval_class, omegaconf.listconfig.ListConfig):
            if len(self.control_eval_class)>2:
                raise Exception("The control_eval_class does not support a list of which length is more than 2!!!")
            self.control_class = self.control_eval_class[0]
            self.control_subclass = self.control_eval_class[1]
        elif self.control_eval_class is None:
            self.control_class = None
            self.control_subclass = None
        else:
            raise Exception("The control_eval_class only support type of str and list!!!")
        
        self.has_delay = -1
        self.best_epoch = None
        self.best_scores = None

    def control(self, epoch, logger, eval_scores=None):
        if eval_scores is None:
            return False
        if self.control_class is not None:
            control_value = None
            for k, v in eval_scores.items():
                if k.lower() == self.control_class.lower():
                    if self.control_subclass is None:
                        control_value = v
                    else:
                        control_value = v[self.control_subclass]
                    break
            if control_value is not None:
                if (control_value - self.best_score)*self.negative_direction>0:
                    self.has_delay+=1
                else:
                    self.has_delay=0
                    self.best_epoch = epoch
                    self.best_scores = eval_scores
                    self.best_score = control_value

                if self.has_delay>=self.delay_epochs:
                    return True
                else:
                    return False
    
    def is_best(self):
        return self.has_delay == 0
    
    def get_best_epoch(self):
        return self.best_epoch
    
    def get_best_score(self):
        return self.best_score
    
    def get_best_scores(self):
        return self.best_scores
    
    def get_control_class(self):
        return self.control_eval_class