import json
import sys

annos = {}
for anno_path in sys.argv[1:]:
    print anno_path
    anno_data = json.load(open(anno_path))
    for a in anno_data:
        if a['name'] not in annos:
            annos[a['name']] = a
        else:
            annos[a['name']]['annotation'].extend(a['annotation'])

json.dump(annos.values(), open('result.json', 'w'))
