import omegaconf
from utils.get_attr import convert_pos_to_dict_element

def auto_args_inherit(configs, 
                      auto_inherit_config,
                      self_word = ["SELF"]):
    x = configs
    levels_list = auto_inherit_config.split('.')
    for tag in levels_list[:-1]:
        x = x[tag]
    
    if isinstance(x[levels_list[-1]], str) \
        and x[levels_list[-1]].split('.')[0] == self_word[0]:
        inherit_obj = x[levels_list[-1]]
        while isinstance(inherit_obj, str) \
            and inherit_obj.split('.')[0] == self_word[0]:
            inherit_obj = convert_pos_to_dict_element(dic={self_word[0]: configs},
                                                      pos=inherit_obj,
                                                      self_dic=configs)
        x[levels_list[-1]] = inherit_obj
    elif isinstance(x[levels_list[-1]], dict) \
        or isinstance(x[levels_list[-1]], omegaconf.dictconfig.DictConfig):
        for k in x[levels_list[-1]].keys():
            auto_args_inherit(configs=configs,
                              auto_inherit_config=auto_inherit_config+'.'+k,
                              self_word=self_word)
    elif isinstance(x[levels_list[-1]], list) \
        or isinstance(x[levels_list[-1]], omegaconf.listconfig.ListConfig):
        for i, k in enumerate(x[levels_list[-1]]):
            if isinstance(k, str) \
                and k.split('.')[0] == self_word[0]:
                inherit_obj = k
                while isinstance(inherit_obj, str) \
                    and inherit_obj.split('.')[0] == self_word[0]:
                    inherit_obj = convert_pos_to_dict_element(dic={self_word[0]: configs},
                                                            pos=inherit_obj,
                                                            self_dic=configs)
                x[levels_list[-1]][i] = inherit_obj