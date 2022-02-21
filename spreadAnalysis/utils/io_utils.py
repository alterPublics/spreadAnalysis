import sys

def load_gexf_nodes(fpath,seq=True):

    nodes = {}
    ncount = 0
    if seq:
        with open(fpath) as infile:
            for line in infile:
                if 'node id=' in str(line):
                    node = str(line).split('node id=')[-1].split('"')[1]
                    nodes[ncount]=node
                    ncount+=1
                if "<edges>" in str(line):
                    break
    return nodes
