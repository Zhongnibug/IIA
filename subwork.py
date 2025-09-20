import omegaconf
from utils.get_attr import no_attr
from utils.get_attr import convert_pos_to_dict_element
from utils.get_attr import from_pos_add_element_to_dict
from utils.get_obj import get_obj_from_str
import os
from datetime import datetime


def subwork(dic,
            subwork_value,
            subwork_key,
            self_dic=None,
            work_detail=None,
            detail_word="DETAIL"):
    
    # Get the name of sub_dict in dic
    sub_dic_name_list = dic.keys()

    if no_attr(subwork_value, "FUNCTION"):
        raise Exception(f"There is no FUNCTION in {subwork_key}!!!")
    else:
        # Load function
        func_str = subwork_value["FUNCTION"]
        while(isinstance(func_str, str) \
              and (func_str.split('.')[0] in sub_dic_name_list)):
            func_str = convert_pos_to_dict_element(
                dic=dic,
                pos=func_str,
                self_dic=None
                )
        if isinstance(func_str, str):
            func = get_obj_from_str(func_str)
        else:
            func = func_str

        # Load input data
        input_data_dict = {}
        if not no_attr(subwork_value, "INPUT"):
            for input_key, input_data in subwork_value["INPUT"].items():
                while(isinstance(input_data, str) \
                    and (input_data.split('.')[0] in sub_dic_name_list)):
                    input_data = convert_pos_to_dict_element(
                        dic=dic,
                        pos=input_data,
                        self_dic=None
                    )
                input_data_dict[input_key] = input_data

        # Get output data
        output_data = func(**input_data_dict)

        # Add output data to dicts
        if (not no_attr(subwork_value, "OUTPUT")) and (not isinstance(subwork_value["OUTPUT"], omegaconf.listconfig.ListConfig)) \
            and (not isinstance(subwork_value["OUTPUT"], list)):
            raise Exception(f"The OUTPUT of {subwork_key} must be list or Null!!!")
        if output_data is None:
            output_data_len = 0
        elif isinstance(output_data, tuple):
            output_data_len = len(output_data)
        else:
            output_data_len = 1
            output_data = [output_data]
        
        if no_attr(subwork_value, "OUTPUT"):
            subwork_out_len = 0
        else:
            subwork_out_len = len(subwork_value["OUTPUT"])
        
        if output_data_len != subwork_out_len:
            raise Exception(f"The OUTPUT of {subwork_key} must be consistent with the output of the real func!!!")
        if not no_attr(subwork_value, "OUTPUT"):
            for i, output_key in enumerate(subwork_value["OUTPUT"]):
                if (output_key == detail_word) and (work_detail is not None):
                    work_detail.update(output_data[i])
                elif output_key != detail_word:
                    from_pos_add_element_to_dict(
                        dic=dic,
                        element=output_data[i],
                        pos=output_key,
                        self_dic=self_dic
                    )
                
    return func.__name__