import time
from collections import defaultdict
import datetime
import torch
import os

class AutoLogPrint:
    def __init__(self, save_path, delimiter="\t", log=True):
        self.save_path = save_path
        if delimiter is None:
            self.delimiter = ""
        else:
            self.delimiter = delimiter
        self.log = log

        self.sub_folder = 'log'
        self.log_path = ""

        self.model_run_logger = None
        self.post_process_logger = None
        self.eval_logger = None

        self.eval_to_post_process = {}

        self.detail_file_name = ''
        self.detail_file_path = ''
        self.detail_file = None

        self.general_file_names = []
        self.general_file_paths = []
        self.general_files = {}

        self.create_folder()
        pass

    def create_folder(self):
        if self.log:
            self.log_path = os.path.join(self.save_path, self.sub_folder)
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)


    def init_model_run_logger(self, header, print_freq, print_summary):
        self.model_run_logger = MetricLogger(delimiter=self.delimiter,
                                             header=header,
                                             print_freq=print_freq,
                                             print_summary=print_summary)
        pass

    
    def init_eval_logger(self, header, print_eval):
        self.eval_logger = InfoLogger(delimiter=self.delimiter,
                                      header=header,
                                      print_info=print_eval,
                                      output_message=self.eval_to_post_process)
        pass

    def init_post_process_logger(self, header, print_post_process):
        self.post_process_logger = InfoLogger(delimiter=self.delimiter,
                                              header=header,
                                              print_info=print_post_process)
        pass

    def log_model_run_info(self, general_file_name):
        if self.log:
            if self.detail_file is not None:
                self.detail_file.write('\n'.join([self.model_run_logger.header,
                                                  self.model_run_logger.summary]))
                self.detail_file.write('\n\n')
            else:
                print(f"NOTE: There is not detail file, the model run summary of {self.model_run_logger.header} does not write into detail file!!!")
            if general_file_name is not None and general_file_name in self.general_file_names:
                self.general_files[general_file_name].write('\n'.join([self.model_run_logger.header,
                                                                       self.model_run_logger.summary]))
                self.general_files[general_file_name].write('\n\n')
            else:
                print(f"NOTE: Specificed general file is None or not created,  the model run summary of {self.model_run_logger.header} does not write into general file {general_file_name}!!!")
        pass

    def log_post_process_info(self):
        if self.log:
            if self.detail_file is not None:
                input_message_keys = self.eval_to_post_process.keys()
                if len(input_message_keys) > 0:
                    for i in range(len(self.post_process_logger.infos)):
                        additional_info = [f" [{k}: {self.eval_to_post_process[k][i]}]" for k in input_message_keys]
                        self.post_process_logger.infos[i] += "".join(additional_info)
                self.detail_file.write('\n'.join([self.post_process_logger.header] + self.post_process_logger.infos))
                self.detail_file.write('\n\n')
            else:
                print(f"NOTE: There is not detail file, the post process info of {self.post_process_logger.header} does not write into detail file!!!")                
        pass

    def log_eval_info(self, general_file_name):
        if self.log:
            if self.detail_file is not None:
                self.detail_file.write('\n'.join([self.eval_logger.header] + self.eval_logger.infos))
                self.detail_file.write('\n\n')
            else:
                print(f"NOTE: There is not detail file, the eval info of {self.eval_logger.header} does not write into detail file!!!")
            if general_file_name is not None and general_file_name in self.general_file_names:
                self.general_files[general_file_name].write('\n'.join([self.eval_logger.header] + self.eval_logger.infos))
                self.general_files[general_file_name].write('\n\n')
            else:
                print(f"NOTE: Specificed general file is None or not created,  the eval info of {self.eval_logger.header} does not write into general file {general_file_name}!!!")
        pass

    def create_detail_file(self, detail_file_name):
        if self.log and detail_file_name != self.detail_file_name:
            self.detail_file_path = os.path.join(self.log_path, f"{detail_file_name}.log")
            self.detail_file = open(self.detail_file_path, "w")
            self.detail_file_name =detail_file_name
            return True
        else:
            return False
        pass

    def save_detail_file(self):
        if self.log and self.detail_file is not None:
            self.detail_file.close()
            self.detail_file = None
            self.detail_file_name = ''
            self.detail_file_path = ''
            self.model_run_to_eval = {}
            self.eval_to_post_process = {}
        pass

    def create_general_file(self, general_file_name):
        if self.log and (general_file_name not in self.general_file_names):
            self.general_file_names.append(general_file_name)
            general_file_path = os.path.join(self.log_path, f"{general_file_name}.log")
            self.general_file_paths.append(general_file_path)
            self.general_files[general_file_name] = open(general_file_path, 'w')
            return True
        else:
            return False
        pass

    def save_general_file(self):
        if self.log and len(self.general_files.keys()) > 0:
            for name, file in self.general_files.items():
                file.close()
            self.general_files = {}
            self.general_file_names = []
            self.general_file_paths = []
        pass

class InfoLogger(object):    
    def __init__(self, delimiter="\t", header=None, print_info=True, output_message=None):
        self.delimiter = delimiter
        if not header:
            self.header = ''
        else:
            self.header = header

        self.print_info = print_info
        self.infos = []
        self.output_message = output_message

    def update(self, info):
        self.infos.append(info)
        if self.print_info:
            print(info)
    
    def output_info(self, name, info):
        if self.output_message is not None:
            self.output_message[name] = info
        else:
            raise Exception("There is no output_message which is defined!!!")

class MetricLogger(object):
    def __init__(self, delimiter="\t", header=None, print_freq=None, print_summary=True):
        self.meters = defaultdict(SimpleValue)
        self.delimiter = delimiter
        if not header:
            self.header = ''
        else:
            self.header = header

        self.print_freq =print_freq
        if self.print_freq is None:
            self.whether_print_iter = False
        else:
            self.whether_print_iter = True

        self.print_summary = print_summary

        self.summary_time = ""
        self.summary_metric = ""
        self.summary = ""

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, fmt):
        self.meters[name] = SimpleValue(fmt=fmt)

    def print_every(self, iterable):        
        i = 0
        start_time = time.time()
        end = time.time()
        iter_time = SimpleValue(fmt='{avg:.4f}')
        data_time = SimpleValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            self.header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
            ])
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if self.whether_print_iter and (i % self.print_freq == 0 or i == len(iterable) - 1):
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.summary_time = '{} Total time: {} ({:.4f} s / it)'.format(
            self.header, total_time_str, total_time / len(iterable))
        if len(self.meters.keys()) > 0:
            self.summary_metric = "Averaged stats: " + self.delimiter.join([f"[{k}:{v.global_avg}]" for k,v in self.meters.items()])
            self.summary = self.summary_time + '\n' + self.summary_metric
        else:
            self.summary = self.summary_time
        
        if self.print_summary:
            print(self.summary)

class SimpleValue(object):
    def __init__(self, fmt=None):
        if fmt is None:
            fmt = "{avg:.4f}"
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.last_value = 0.0
        self.num = 0

    def update(self, value, num=1):
        self.count += num
        self.total += value
        self.last_value = value
        self.num = num

    @property
    def avg(self):
        return self.last_value / self.num
    
    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def value(self):
        return self.last_value

    def __str__(self):
        return self.fmt.format(
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value)       
    
