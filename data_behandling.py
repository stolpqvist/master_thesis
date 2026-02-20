import os
def gen(x):
    with open(x, 'r') as file:
        for line in file.read():
            yield line

def extractor(file):
    in_string = False
    for line in gen(file):
        
        if in_string:
            if line.endswith(('"', "'")):
                in_string=False
            continue
        
        if line.startswith(('"', "'")):
            in_string = True
            continue

