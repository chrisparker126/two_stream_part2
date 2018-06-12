import re 
import csv
import itertools
from matplotlib import pyplot as plt
import numpy as np

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


def get_train_data_opt_flow(listPath):
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

def get_test_data_opt_flow(listPath, classIndexPath):
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')