import torch
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def tuple_cat(tuple_l):
    res = []
    for i in range(len(tuple_l[0])):
        res.append(torch.cat([t[i] for t in tuple_l], dim=0))
    return tuple(res)

def tuple_stack(tuple_l):
    res = []
    for i in range(len(tuple_l[0])):
        res.append(torch.stack([t[i] for t in tuple_l]))
    return tuple(res)

import subprocess
def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
