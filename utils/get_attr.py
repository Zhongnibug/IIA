from omegaconf import OmegaConf
import omegaconf
import numpy as np

def no_attr(father, attr):
    if father is None:
        return True
    if isinstance(father, dict):
        if attr not in father:
            return True
    else:
        if not hasattr(father, attr):
            return True
    if father[attr] is None:
        return True
    return False

def diff_default_conf(default_conf, conf):
    diffs = []

    for k, v in default_conf.items():
        if no_attr(conf, k) and not no_attr(default_conf, k):
            diffs.append([k])
        elif (isinstance(default_conf[k], dict) or \
              isinstance(default_conf[k], omegaconf.dictconfig.DictConfig)) and \
                (isinstance(conf[k], dict) or \
                 isinstance(conf[k], omegaconf.dictconfig.DictConfig)) and \
            default_conf[k] != conf[k]:
            sub_diffs = diff_default_conf(default_conf[k], conf[k])
            diffs += [[k] + sub_k for sub_k in sub_diffs]
        elif default_conf[k] != conf[k]:
            diffs.append([k])

    for k, v in conf.items():
        if no_attr(default_conf, k) and not no_attr(conf, k):
            diffs.append([k])

    return diffs

def compare_has_conf_is_equal(default_conf, conf):
    '''
    Traverse through conf to check if all the configurations in conf match those in default_conf, 
    without considering configurations that are present in one but not in the other.
    strict_flag has the more strict requise that default_conf must have all configurations in conf.
    ''' 

    flag = True
    strict_flag = True
    for k, v in conf.items():
        if no_attr(default_conf, k):
            strict_flag = False
            continue
        elif (isinstance(default_conf[k], dict) or \
              isinstance(default_conf[k], omegaconf.dictconfig.DictConfig)) and \
                (isinstance(conf[k], dict) or \
                 isinstance(conf[k], omegaconf.dictconfig.DictConfig)):
            flag, strict_flag = compare_has_conf_is_equal(default_conf[k], conf[k])
        else:
            flag = (default_conf[k] == conf[k])
        if not flag:
            return flag, flag & strict_flag
    return flag, flag & strict_flag


def convert_all_f64_to_f32(config):
    if isinstance(config, dict):
        for k, v in config.items():
            config[k] = convert_all_f64_to_f32(v)
    elif isinstance(config, list):
        config = [convert_all_f64_to_f32(i) for i in config]
    elif isinstance(config, np.float64):
        return float(config)
    return config

def convert_pos_to_dict_element(dic, pos, self_dic=None):
    if not isinstance(pos, str):
        raise Exception("The pos arg must be str!!!")
    
    levels_list = pos.split('.')

    if levels_list[0] not in dic:
        x = self_dic
    else:
        x = dic

    for tag in levels_list:
        try:
            x = x[tag]
        except:
            x = pos
            break
    return x

def deep_update(d, u):
    for k, v in u.items():
        if (isinstance(v, dict) or isinstance(v, omegaconf.dictconfig.DictConfig)) and\
              k in d and \
                (isinstance(d[k], dict) or isinstance(d[k], omegaconf.dictconfig.DictConfig)):
            deep_update(d[k], v)
        else:
            d[k] = v

def from_pos_add_element_to_dict(dic, element, pos, self_dic=None):

    if not isinstance(pos, str):
        raise Exception("The pos arg must be str!!!")
    levels_list = pos.split('.')
    change_tag = levels_list[-1]

    if levels_list[0] not in dic and self_dic is None:
        raise Exception("The first level of the pos is not the key of dict and self dict is None!!!")
    elif levels_list[0] not in dic:
        x = self_dic
    else:
        x = dic

    for i, tag in enumerate(levels_list[:-1]):
        if no_attr(x, tag):
            temp_e = {}
            temp_e[levels_list[-1]] = element
            for j in range(len(levels_list)-2, i, -1):
                temp_dic = {}
                temp_dic[levels_list[j]] = temp_e
                temp_e = temp_dic
            element = temp_e
            change_tag = tag
            break
        else:
            x = x[tag]
            change_tag = levels_list[i+1]
    if isinstance(element, dict) or isinstance(element, omegaconf.dictconfig.DictConfig):
        y = x.get(change_tag, {})
        deep_update(y, element)
        x[change_tag] = y
    else:
        x[change_tag] = element

def from_pos_del_and_free(dic, pos, self_dic=None):

    if not isinstance(pos, str):
        raise Exception("The pos arg must be str!!!")
    levels_list = pos.split('.')
    change_tag = levels_list[-1]

    if levels_list[0] not in dic and self_dic is None:
        raise Exception("The first level of the pos is not the key of dict and self dict is None!!!")
    elif levels_list[0] not in dic:
        x = self_dic
    else:
        x = dic

    for i, tag in enumerate(levels_list[:-1]):
        if no_attr(x, tag):
            raise Exception(f"There is no {pos}!!!")
        else:
            x = x[tag]
            change_tag = levels_list[i+1]

    del x[change_tag]
    
def add_omgaconf_from_other(configs,
                            other_configs,
                            no_add_keys):
    for addkey, addvalue in other_configs.items():
        if addkey not in no_add_keys:
            configs[addkey] = addvalue