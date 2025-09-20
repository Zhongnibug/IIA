import os
import pickle

def get_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab =  pickle.load(f)
    return vocab    

class BaseVocabulary:
    def __init__(self, special_vocabulary_list):
        self.i2w = {}
        self.w2i = {}
        self.length = 0
        self.special_vocabulary_list = special_vocabulary_list

        self.init_dicts()
        pass

    def init_dicts(self):
        if self.special_vocabulary_list is not None:
            if not isinstance(self.special_vocabulary_list, list):
                raise Exception("Special vocabulary list must be list!!!")
            for word in self.special_vocabulary_list:
                self.i2w[self.length] = word
                self.w2i[word] = self.length
                self.length+=1

    def build(self):
        raise Exception("Children class does not have build function!!!")
                        
    def index_to_word(self, index):
        return self.i2w[index]
    
    def word_to_index(self, word):
        return self.w2i[word]
    
    def index_list_to_string(self, index_list):
        return " ".join([self.index_to_word(index) for index in index_list])

    def string_to_index_list(self, string):
        return [self.word_to_index(word) for word in string.split()]

class SimpleVocabulary(BaseVocabulary):
    def __init__(self, special_vocabulary_list):
        super(SimpleVocabulary, self).__init__(special_vocabulary_list)
        pass

    def build(self, captions_dict):
        for _,captions_list in captions_dict.items():
            for caption in captions_list:
                for word in caption.split():
                    if word not in self.w2i:
                        self.i2w[self.length] = word
                        self.w2i[word] = self.length
                        self.length+=1
        pass