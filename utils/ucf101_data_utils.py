import re 
import csv

def get_train_data(listPath):
# load Id list and labels 

    train_list = list()

    with open(listPath, 'r') as f:
        reader = csv.reader(f)
        train_list = list(reader)

        labels = [int(label[0].split(' ')[1]) for label in train_list ]
    IDs = [label[0].split(' ')[0] for label in train_list ]
    # IDs
    IDs = [id.split('/')[1].rstrip('.avi') for id in IDs ]

    labels = dict(zip(IDs, labels))
    return (IDs, labels)

def get_test_data(listPath, classIndexPath):
# load Id list and labels 
    
    train_list = None
    class_labels = None
    with open(listPath, 'r') as f:
        reader = csv.reader(f)
        train_list = list(reader)

    with open(classIndexPath, 'r') as f:
        reader = csv.reader(f)
        class_labels = list(reader)
    
    ucf_class = [c[0].split(' ')[1] for c in class_labels]
    ucf_labels = [int(c[0].split(' ')[0]) for c in class_labels]
    
    ucf_dict = dict(zip(ucf_class, ucf_labels))
    
    IDs = [id_[0].split('/')[1].rstrip('.avi') for id_ in train_list ]
    labels = dict()
    
    p = re.compile('v\_([a-zA-Z]*)\_g\w*') 
    for id_ in IDs:
        f = p.match(id_)
        if f:
            labels[id_] = ucf_dict[f[1]]
        else:
            Exception("failed")
    return IDs, labels