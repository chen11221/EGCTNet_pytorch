
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = r'Z:\cj\datasourse\LEVIR-CD-256'
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir = r'E:\bianhuajiance\WHU'
        elif data_name == 'quick_start_LEVIR':
            self.root_dir = './samples_LEVIR/'
        elif data_name == 'quick_start_WHU':
            self.root_dir = './samples_WHU/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self

