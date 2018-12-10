import in_place
import os
from builtins import str

for path, subdirs, files in os.walk('C:/Users/cdilg/Documents/NEU/CS6120/reverb jar/data'):
    for name in files:
        with in_place.InPlace(os.path.join(path, name), encoding='utf-8') as f:
            for line in f:
                line.replace('\t', '    ')
                f.write(line)
            
