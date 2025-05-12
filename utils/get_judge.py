from omegaconf import OmegaConf
import omegaconf

def get_caption_limit_judge(caption_limit_len):
    if isinstance(caption_limit_len, int):
        caption_limit_judge = lambda x: x<caption_limit_len
    elif isinstance(caption_limit_len, list) \
        or isinstance(caption_limit_len, omegaconf.listconfig.ListConfig):
        if len(caption_limit_len)<2:
            raise Exception("If caption_limit_len is a list, the length of it must be bigger than 1!!!")
        elif isinstance(caption_limit_len[0], int) and isinstance(caption_limit_len[1], int):
            caption_limit_judge = lambda x: caption_limit_len[0]<=x<caption_limit_len[1]                                                                       
        elif isinstance(caption_limit_len[1], int):
            caption_limit_judge = lambda x: x<caption_limit_len[1]
        elif isinstance(caption_limit_len[0], int):
            caption_limit_judge = lambda x: x>=caption_limit_len[0]
        else:
            caption_limit_judge = lambda x: True
    else:
        caption_limit_judge = lambda x: True

    return caption_limit_judge