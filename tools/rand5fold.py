import random
from random import shuffle
import os
from tools.pargs import pargs
args = pargs()
random.seed(args.seed)
cwd=os.getcwd()
pheme_event = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege']

def load5foldData():
    if args.dataset=="PHEME":
        dataset = args.dataset
        dataset_tree = args.dataset_tree
        dataset_label = args.dataset_label
        fold0_x_test_FIN, fold1_x_test_FIN, fold2_x_test_FIN, fold3_x_test_FIN, fold4_x_test_FIN = [], [], [], [], []
        fold0_x_train_FIN, fold1_x_train_FIN, fold2_x_train_FIN, fold3_x_train_FIN, fold4_x_train_FIN = [], [], [], [], []
        for event in pheme_event:
            labelPath = os.path.join(f"../../data/{dataset}/{dataset_label}_{event}.txt")
            print(f"5-fold for {dataset_label}_{event}:")
            F, T = [], []
            l1 = l2 = 0
            labelDic = {}
            for line in open(labelPath):
                line = line.rstrip()
                eid, label = line.split(' ')[0], line.split(' ')[1]
                if os.path.exists(os.path.join('..','..', 'data', f'{args.dataset}graph',eid+".npz")):
                    labelDic[eid] = int(label)
                    if labelDic[eid] == 0:
                        F.append(eid)
                        l1 += 1
                    if labelDic[eid] == 1:
                        T.append(eid)
                        l2 += 1
            print(len(labelDic))
            print(l1, l2)
            random.shuffle(F)
            random.shuffle(T)

            fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
            fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
            leng1 = int(l1 * 0.2)
            leng2 = int(l2 * 0.2)
            fold0_x_test.extend(F[0:leng1])
            fold0_x_test.extend(T[0:leng2])
            fold0_x_train.extend(F[leng1:])
            fold0_x_train.extend(T[leng2:])
            fold0_x_test_FIN.extend(fold0_x_test)
            fold0_x_train_FIN.extend(fold0_x_train)

            fold1_x_train.extend(F[0:leng1])
            fold1_x_train.extend(F[leng1 * 2:])
            fold1_x_train.extend(T[0:leng2])
            fold1_x_train.extend(T[leng2 * 2:])
            fold1_x_test.extend(F[leng1:leng1 * 2])
            fold1_x_test.extend(T[leng2:leng2 * 2])
            fold1_x_test_FIN.extend(fold1_x_test)
            fold1_x_train_FIN.extend(fold1_x_train)

            fold2_x_train.extend(F[0:leng1 * 2])
            fold2_x_train.extend(F[leng1 * 3:])
            fold2_x_train.extend(T[0:leng2 * 2])
            fold2_x_train.extend(T[leng2 * 3:])
            fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
            fold2_x_test.extend(T[leng2 * 2:leng2 * 3])
            fold2_x_test_FIN.extend(fold2_x_test)
            fold2_x_train_FIN.extend(fold2_x_train)

            fold3_x_train.extend(F[0:leng1 * 3])
            fold3_x_train.extend(F[leng1 * 4:])
            fold3_x_train.extend(T[0:leng2 * 3])
            fold3_x_train.extend(T[leng2 * 4:])
            fold3_x_test.extend(F[leng1 * 3:leng1 * 4])
            fold3_x_test.extend(T[leng2 * 3:leng2 * 4])
            fold3_x_test_FIN.extend(fold3_x_test)
            fold3_x_train_FIN.extend(fold3_x_train)

            fold4_x_train.extend(F[0:leng1 * 4])
            fold4_x_train.extend(F[leng1 * 5:])
            fold4_x_train.extend(T[0:leng2 * 4])
            fold4_x_train.extend(T[leng2 * 5:])
            fold4_x_test.extend(F[leng1 * 4:leng1 * 5])
            fold4_x_test.extend(T[leng2 * 4:leng2 * 5])
            fold4_x_test_FIN.extend(fold4_x_test)
            fold4_x_train_FIN.extend(fold4_x_train)

        fold0_test = list(fold0_x_test_FIN)
        shuffle(fold0_test)
        fold0_train = list(fold0_x_train_FIN)
        shuffle(fold0_train)
        fold1_test = list(fold1_x_test_FIN)
        shuffle(fold1_test)
        fold1_train = list(fold1_x_train_FIN)
        shuffle(fold1_train)
        fold2_test = list(fold2_x_test_FIN)
        shuffle(fold2_test)
        fold2_train = list(fold2_x_train_FIN)
        shuffle(fold2_train)
        fold3_test = list(fold3_x_test_FIN)
        shuffle(fold3_test)
        fold3_train = list(fold3_x_train_FIN)
        shuffle(fold3_train)
        fold4_test = list(fold4_x_test_FIN)
        shuffle(fold4_test)
        fold4_train = list(fold4_x_train_FIN)
        shuffle(fold4_train)

        return list(fold0_test), list(fold0_train), \
               list(fold1_test), list(fold1_train), \
               list(fold2_test), list(fold2_train), \
               list(fold3_test), list(fold3_train), \
               list(fold4_test), list(fold4_train)

    else:
        dataset = args.dataset
        dataset_tree = args.dataset_tree
        dataset_label = args.dataset_label
        labelPath = os.path.join(f"../../data/{dataset}/{dataset_label}.txt")
        print(f"Load 5-fold for {dataset}:")
        F, T = [], []
        l1 = l2 = 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            eid,label = line.split(' ')[0], line.split(' ')[1]
            labelDic[eid] = int(label)
            if labelDic[eid]==0:
                F.append(eid)
                l1 += 1
            if labelDic[eid]==1:
                T.append(eid)
                l2 += 1
        print(len(labelDic))
        print(l1, l2)
        random.shuffle(F)
        random.shuffle(T)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        fold0_x_test.extend(F[0:leng1])
        fold0_x_test.extend(T[0:leng2])
        fold0_x_train.extend(F[leng1:])
        fold0_x_train.extend(T[leng2:])

        fold1_x_train.extend(F[0:leng1])
        fold1_x_train.extend(F[leng1 * 2:])
        fold1_x_train.extend(T[0:leng2])
        fold1_x_train.extend(T[leng2 * 2:])
        fold1_x_test.extend(F[leng1:leng1 * 2])
        fold1_x_test.extend(T[leng2:leng2 * 2])

        fold2_x_train.extend(F[0:leng1 * 2])
        fold2_x_train.extend(F[leng1 * 3:])
        fold2_x_train.extend(T[0:leng2 * 2])
        fold2_x_train.extend(T[leng2 * 3:])
        fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
        fold2_x_test.extend(T[leng2 * 2:leng2 * 3])

        fold3_x_train.extend(F[0:leng1 * 3])
        fold3_x_train.extend(F[leng1 * 4:])
        fold3_x_train.extend(T[0:leng2 * 3])
        fold3_x_train.extend(T[leng2 * 4:])
        fold3_x_test.extend(F[leng1 * 3:leng1 * 4])
        fold3_x_test.extend(T[leng2 * 3:leng2 * 4])

        fold4_x_train.extend(F[0:leng1 * 4])
        fold4_x_train.extend(F[leng1 * 5:])
        fold4_x_train.extend(T[0:leng2 * 4])
        fold4_x_train.extend(T[leng2 * 5:])
        fold4_x_test.extend(F[leng1 * 4:leng1 * 5])
        fold4_x_test.extend(T[leng2 * 4:leng2 * 5])

        fold0_test = list(fold0_x_test)
        shuffle(fold0_test)
        fold0_train = list(fold0_x_train)
        shuffle(fold0_train)
        fold1_test = list(fold1_x_test)
        shuffle(fold1_test)
        fold1_train = list(fold1_x_train)
        shuffle(fold1_train)
        fold2_test = list(fold2_x_test)
        shuffle(fold2_test)
        fold2_train = list(fold2_x_train)
        shuffle(fold2_train)
        fold3_test = list(fold3_x_test)
        shuffle(fold3_test)
        fold3_train = list(fold3_x_train)
        shuffle(fold3_train)
        fold4_test = list(fold4_x_test)
        shuffle(fold4_test)
        fold4_train = list(fold4_x_train)
        shuffle(fold4_train)

        return list(fold0_test),list(fold0_train),\
               list(fold1_test),list(fold1_train),\
               list(fold2_test),list(fold2_train),\
               list(fold3_test),list(fold3_train),\
               list(fold4_test), list(fold4_train)
