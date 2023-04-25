import numpy as np

def cpp_type(t: np.dtype):
    if t == np.float32:
        return 'float'
    elif t == np.int32:
        return 'int32_t'
    elif t == np.int64:
        return 'int64_t'
    elif t == np.bool_:
        return 'bool'
    else:
        raise ValueError(f'Unsupported dtype {t}')

class CodeWriter:
    def __init__(self):
        self.code_str = ''
        self.intend = 0

    def block_start(self):
        self.code_str += '    ' * self.intend + '{\n'
        self.intend += 1
    
    def block_end(self):
        self.intend -= 1
        self.code_str += '    ' * self.intend + '}\n'

    def write(self, code_str: str):
        code = code_str.splitlines()
        for line in code:
            self.code_str += '    ' * self.intend + line + '\n'
    
    def wl(self, code_str: str):
        self.write(code_str + '\n')
    
    def get_code(self):
        return self.code_str
