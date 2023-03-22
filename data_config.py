
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
        elif data_name == 'WHU-512-100':
            self.label_transform = "norm"
            self.root_dir = r'E:\bianhuajiance\database\WHU-CD-512-100'
        elif data_name == 'WHU-512-0':
            self.label_transform = "norm"
            self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-0'
        elif data_name == 'WHU-512-10':
            self.label_transform = "norm"
            self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-10'
        elif data_name == 'WHU-512-20':
            self.label_transform = "norm"
            self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-20'
        elif data_name == 'WHU-512-30':
            self.label_transform = "norm"
            self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-30'
        elif data_name == 'WHU-512-30-only':
            self.label_transform = "norm"
            self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-30-only'
        elif data_name == 'WHU-512-40':
            self.label_transform = "norm"
            self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-40'
        elif data_name == 'WHU-512-40-only':
            self.label_transform = "norm"
            self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-40-only'
        elif data_name == 'WHU-512-50':
            self.label_transform = "norm"
            self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-50'
        elif data_name == 'WHU-512-50-only':
            self.label_transform = "norm"
            self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-50-only'
        elif data_name == 'quick_start_LEVIR':
            self.root_dir = './samples_LEVIR/'
        elif data_name == 'quick_start_WHU':
            self.root_dir = './samples_WHU/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self

