#!/usr/bin/env python
# coding: utf-8
import numpy as np
import glob, time
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import EnglishStemmer

FOLDER = '../enron1/'
HAM_FOLDER = 'ham/'
SPAM_FOLDER = 'spam/'

HAM_LIST = glob.glob(FOLDER + HAM_FOLDER + '*.txt')
SPAM_LIST = glob.glob(FOLDER + SPAM_FOLDER + '*.txt')

class Classifier:
    
    HAM = 0
    SPAM = 1

    def __init__(self, ham_list, spam_list, f_train = None):
        self.ham_list = ham_list # file list of ham emails
        self.spam_list = spam_list # file list of spam emails

        self.N_HAM = np.size(ham_list) # number of total ham emails
        self.N_SPAM = np.size(spam_list) # number of total spam emails
        self.N = np.asarray([self.N_HAM, self.N_SPAM])
        self.label = np.asarray([self.HAM]* self.N_HAM + [self.SPAM]* self.N_SPAM)
        
        # container for vocabulary list
        self.vocab = None
        self.nvocab = 0
        
        self.f_train = 0.8 if not f_train else f_train # fraction of train in total, default to be 0.8
        # num of train docs in ham and spam folder
        self.N_TRAIN = np.asarray([int(np.floor(self.N_HAM * self.f_train)), 
                                      int(np.floor(self.N_SPAM * self.f_train))]) 
        self.N_TEST = self.N - self.N_TRAIN
        self.train_X, self.train_label, self.test_X, self.test_label = self.vectorize(self.f_train)
        
        self.result = None
    
    def vectorize(self, f_train = None):
        start_time = time.time()
        if f_train is not None:
            self.f_train = f_train # else f_train = 0.8 by default
            # [number of ham in train, number of spam in train]
            self.N_TRAIN = np.asarray([int(np.floor(self.N_HAM * self.f_train)),
                                          int(np.floor(self.N_SPAM * self.f_train))])
            # [number of ham in test, number of spam in test]
            self.N_TEST = self.N - self.N_TRAIN
        print('vectorizing the emails...')
        print('%s %% of emails are used for train...' % (self.f_train * 100))

        # word stemming
        # we are filtering out:
        #    numbers, words shorter than 3 letters, words appeared less than 5 times in total,
        #    words that appeared in 95% of the emails
        pre = CountVectorizer(input = 'filename', decode_error = 'ignore', 
                              token_pattern = u'(?ui)\\b\\w*[a-z]+\\w{3,}\\b', max_df = 0.95, min_df = 5)
        pre_X = pre.fit_transform(self.ham_list[:self.N_TRAIN[self.HAM]] + 
                                  self.spam_list[:self.N_TRAIN[self.SPAM]]).toarray()
        
        # get the vocabulary list from train data
        prevocab = pre.get_feature_names()
#         stemmer = EnglishStemmer()
        stemmed = [EnglishStemmer().stem(w) for w in prevocab]
        self.vocab = np.unique(stemmed)
        self.nvocab = np.size(self.vocab)
        
        # train data vectorized with our vocabulary
        train = CountVectorizer(input = 'filename', decode_error = 'ignore', vocabulary = self.vocab)
        self.train_X = train.fit_transform(self.ham_list[:self.N_TRAIN[self.HAM]] + 
                                                 self.spam_list[:self.N_TRAIN[self.SPAM]]).toarray()

        # test data vectorized with our vocabulary
        test = CountVectorizer(input = 'filename', vocabulary = self.vocab, decode_error = 'ignore')
        self.test_X = test.fit_transform(self.ham_list[-self.N_TEST[self.HAM]:] + 
                                               self.spam_list[-self.N_TEST[self.SPAM]:]).toarray()
        
        # create the label arrays
        self.train_label = np.asarray([self.HAM] * self.N_TRAIN[self.HAM] + 
                                         [self.SPAM] * self.N_TRAIN[self.SPAM])
        self.test_label = np.asarray([self.HAM] * self.N_TEST[self.HAM] + 
                                        [self.SPAM] * self.N_TEST[self.SPAM])
        
        print('vectorizing done! it took %.2f s' % (time.time() - start_time))
        return self.train_X, self.train_label, self.test_X, self.test_label
    
    def get_train(self):
        '''return the input matrix and label for train set
            Output:
                trainging_X, train_label'''
        return self.train_X, self.train_label
    
    def get_test(self):
        '''return the input matrix and label for test set
            Output:
                test_X, test_label'''
        return self.test_X, self.test_label
    
    def split_ham_spam(self, X, label):
        '''split X based on label HAM/SPAM'''
        return X[np.where(label == self.HAM)], X[np.where(label == self.SPAM)]
                
    def accuracy(self, result = None):
        if result is None:
            if self.result is None:
                print('no results recorded!')
                return np.nan
            else:
                return np.mean(self.result == self.test_label) # num of correct predictions / total
        else:
            return np.mean(result == self.test_label) # num of correct predictions / total
            

    def naive_bayes(self, f_train = None):
        if f_train is not None:
            # re-vectorize the data
            self.vectorize(f_train)
        # we use the multinomial naive bayes model from 
        # https://web.stanford.edu/class/cs124/lec/naivebayes.pdf
        def get_prior():
            '''get the prior of for the Naive Bayes method which will be
            [fraction of ham emails in train set, 
            fraction of spam emails in train set]'''
            prior = self.N_TRAIN / self.N_TRAIN.sum()
            return prior

        def get_conditionals():
            '''get the conditionals of for the Naive Bayes method with some smoothing'''
            # split the traning data by label
            train_ham, train_spam = self.split_ham_spam(self.train_X, self.train_label)

            # conditionals with Laplace smoothing
            con_ham = (train_ham.sum(axis = 0) + 1) / (train_ham.sum() + self.nvocab)
            con_spam = (train_spam.sum(axis = 0) + 1) / (train_spam.sum() + self.nvocab)
            conditionals = np.asarray([con_ham, con_spam])
            return conditionals

        print('cross validating...')
        start_time = time.time()

        prior = get_prior()
        conditionals = get_conditionals()
        # start applying labels to our test data!
        self.result = np.empty(self.N_TEST.sum()) # the results of our classifier
        for i in np.arange(self.N_TEST.sum()):
            # use log likelihood for easier calculation
            loglike_ham = np.dot(np.log(conditionals[self.HAM]), self.test_X[i]) + np.log(prior[self.HAM])
            loglike_spam = np.dot(np.log(conditionals[self.SPAM]), self.test_X[i]) + np.log(prior[self.SPAM])
            self.result[i] = self.HAM if loglike_ham > loglike_spam else self.SPAM
        print('test took %.2f s' % (time.time() - start_time))
        return self.result

    def nearest_neighbor(self, f_train = None):
        if f_train != None:
            # re-vectorize the data
            self.vectorize(f_train)

        print('running classifier...')
        start_time = time.time()

        def calculate_l1_distance(train_row, test_row):
            diff_row = np.subtract(train_row, test_row) # find element wise difference
            diff_row = np.absolute(diff_row) # take absolute value of differences
            distance = np.sum(diff_row) # sum the distances
            return distance


        def calculate_l2_distance(train_row, test_row):
            diff_row = np.subtract(train_row, test_row)
            diff_row = np.square(diff_row)
            distance = np.sum(diff_row)
            return np.sqrt(distance)


        def calculate_linf_distance(train_row, test_row):
            diff_row = np.subtract(train_row, test_row)
            diff_row = np.absolute(diff_row)
            return np.amax(diff_row)

        predicted_label_l1 = np.empty(shape = (len(self.test_X), 1), dtype = int)
        predicted_label_l2 = np.empty(shape = (len(self.test_X), 1), dtype = int)
        predicted_label_linf = np.empty(shape = (len(self.test_X), 1), dtype = int)
        for test_row, i in zip(self.test_X, range(len(self.test_X))):
            row_distance_l1 = np.empty(shape = (len(self.train_X), 1), dtype = int)
            row_distance_l2 = np.empty(shape = (len(self.train_X), 1), dtype = int)
            row_distance_linf = np.empty(shape = (len(self.train_X), 1), dtype = int)
            for train_row, j in zip(self.train_X, range(len(self.train_X))):
                distance_l1 = calculate_l1_distance(train_row, test_row)
                distance_l2 = calculate_l2_distance(train_row, test_row)
                distance_linf = calculate_linf_distance(train_row, test_row)
                row_distance_l1[j] = distance_l1 # array of distances for each test row
                row_distance_l2[j] = distance_l2
                row_distance_linf[j] = distance_linf
                # print("test row:", test_row, "  | label: ", self.test_X_label[i])
                # print("train row:", train_row, " | label: ", self.train_label[j])
                # print("dist sum: ", distance)
            min_dist_index_l1 = np.argmin(row_distance_l1) # min distance's index in array of distances
            min_dist_index_l2 = np.argmin(row_distance_l2)
            min_dist_index_linf = np.argmin(row_distance_linf)

            predicted_label_l1[i] = self.train_label[min_dist_index_l1]
            predicted_label_l2[i] = self.train_label[min_dist_index_l2]
            predicted_label_linf[i] = self.train_label[min_dist_index_linf]
            # print("-----------------------")
            # print("min dist: ", np.amin(row_distance))
            # print("index of min: ", np.argmin(row_distance))
            # print("predicted label: ", predicted_label[i])
            # print("-----------------------\n")
            self.result = [predicted_label_l1.flatten(), 
                           predicted_label_l2.flatten(), 
                           predicted_label_linf.flatten()]
        print('test took %.2f s' % (time.time() - start_time))
        return self.result
    
    def decision_tree(self, f_train = None):
        if f_train is not None:
            # re-vectorize the data
            self.vectorize(f_train)
        
        print('running classifier...')
        start_time = time.time()
       
        class TreeNode:
            def __init__(self, idx):
                self.idx = idx
#                 print('a new node at index %d' % (self.idx, ))
                self.value = np.nan
                self.left = None
                self.right = None
                
        class ListNode:
            def __init__(self, idx):
                self.idx = idx
                self.prev = None
                self.next = None
                
            def is_head(self):
                return True if self.prev is None else False
            
            def is_tail(self):
                return True if self.next is None else False

        class List:
            def __init__(self, idx_list):
                # Nodes for List
                self.idx_list = idx_list
                self.head = ListNode(idx_list[0])
                self.length = 1
                self.construct()

            def construct(self):
                current_Node = self.head
                for i in np.arange(1, np.size(self.idx_list)):
#                     print('the %d node is constructed with index %d' % (i, self.idx_list[i]))
                    current_Node.next = ListNode(self.idx_list[i])
                    current_Node.next.prev = current_Node
                    current_Node = current_Node.next
                    self.length += 1

            def pop(self, current_node): 
                if self.length < 1:
                    print('the size of list is %d. there is no nodes to pop.' % self.length)
                elif self.length > 1:
                    if current_node.is_head():
                        next_n = current_node.next
                        self.head = next_n
                        next_n.prev = None
                    elif current_node.is_tail():
                        last_n = current_node.prev
                        last_n.next = None
                    else:
                        last_n = current_node.prev
                        next_n = current_node.next
                        last_n.next = next_n
                        next_n.prev = last_n
                else: # list size = 1, this node is the only item in the List
                    current_node = None
                self.length -= 1

            def is_empty(self):
                return True if self.length < 1 else False  

        def build_tree(rows):
            '''build decision tree with the train data
            Parameters:
                rows: 1darray, row numbers that goes into this node; for root node, this is 
                np.arange(total_row_number_of_train_data); each row can be seen as one data
                point'''
            # if the region contains less data points than LEAF_SIZE, make this into a leaf
            # if there is no more words in the list
            if np.size(rows) < LEAF_SIZE or arg_list.is_empty():
#                 print('this leaf has %d data' % np.size(rows))
                # the leaf is a label HAM or SPAM, depending only on the label in the leaf 
                # if fraction of HAM is higher, leaf is HAM and vice versa
                if (self.train_label[rows] == self.HAM).mean() > 0.5: 
                    return self.HAM
                else:
                    return self.SPAM
            else:
                # get the head of arg_list, which is the position of the word with highest
                # frequency difference in HAM vs. SPAM
                current_node = arg_list.head
                # the column slice of train_X at position arg_list.head.idx, the size is
                # the same as the total number of data points(total number of rows)
                col = self.train_X[:, current_node.idx]
                # get the data points and their corresponding labels
                # the list of frequency for word[idx]
                node_data, node_labels = col[rows], self.train_label[rows]
                current_value = unc_min(node_data, node_labels) # find the lowest gini index value for split
                l_rows = rows[np.where(node_data <= current_value)] # split by value
                r_rows = rows[np.where(node_data > current_value)]
                while np.size(l_rows) * np.size(r_rows) == 0:
                    if current_node.is_tail():
#                         print('reached tail')
                        break
                    else:
                        current_node = current_node.next
    #                     print('looking for spliting point at %d' % idx)
                        col = self.train_X[:, current_node.idx] # the column turned into array
                        node_data = col[rows]
                        node_labels = self.train_label[rows]
                        # find the lowest gini index value for split
                        current_value = unc_min(node_data, node_labels)
                        l_rows = rows[np.where(node_data <= current_value)] # split by value
                        r_rows = rows[np.where(node_data > current_value)]
                # either the current node is tail, or the split is good enough
                # arg_list not empty
                new_N = TreeNode(current_node.idx)
                new_N.value = current_value
                arg_list.pop(current_node)
                if np.size(l_rows) == 0:
                    new_N.left = self.SPAM if (self.train_label[r_rows] == self.HAM).mean() > 0.5 else self.HAM
                    new_N.right = build_tree(r_rows)
                    return new_N
                elif np.size(r_rows) == 0:
                    new_N.right = self.SPAM if (self.train_label[l_rows] == self.HAM).mean() > 0.5 else self.HAM
                    new_N.left = build_tree(l_rows)
                    return new_N
                else:
                    new_N.left = build_tree(l_rows)
                    new_N.right = build_tree(r_rows)
                    return new_N
                    
        def unc_min(data, label):
            '''return the frequency value of minimum uncertainty given an array 
            of word freq and associated label
            Parameters:
            col: 1darray of word frequencies at data points
            label: 1darray of labels corresponding to these data points'''
#             f_max, f_min = np.min(data[np.nonzero(data)]), np.max(data[np.nonzero(data)])
            f_max, f_min = np.max(data), np.min(data) # the largest and the smallest value/freq in column
            # if the maximun equals the minimum -> all values are equal
            # return that value to split
            if f_max == f_min: 
                return f_max 
            else: # if not all elements are zero
                split_value = np.linspace(f_min, f_max, num = SPLIT) # the values of diff split
                unc = uncertainty(split_value, data, label) # list of gini idx at diff split
                return split_value[unc.argmin()] # return the value for min uncertainty

        # Calculate the uncertainty for a split dataset
        def _uncertainty(cut, data, label):
            # labels for left and right nodes
            l_label, r_label = label[np.where(data <= cut)], label[np.where(data > cut)]
            l_len, r_len = np.size(l_label), np.size(r_label)
            
            unc = 0.0        
            if l_len > 0:
                l_p_ham = (l_label == self.HAM).mean()
                l_p_spam = (l_label == self.SPAM).mean()
                unc += (1 - (l_p_ham**2 + l_p_spam**2)) * (l_len / (l_len + r_len))

            if r_len > 0:
                r_p_ham = (r_label == self.HAM).mean()
                r_p_spam = (r_label == self.SPAM).mean()
                unc += (1 - (r_p_ham**2 + r_p_spam**2)) * (r_len / (l_len + r_len))
            return unc
        # vectorize this function so that it can take ndarrays as the first argument
        uncertainty = np.vectorize(_uncertainty, excluded = [1, 2])
        
        def classify(case, N):
            if N == self.HAM:
#                 print('this is HAM')
                return self.HAM
            elif N == self.SPAM:
#                 print('this is SPAM')
                return self.SPAM
            else:
                if case[N.idx] <= N.value:
#                     print('at %d (%s) freq is smaller than %f' % (N.idx, self.vocab[N.idx], N.value))
                    return classify(case, N.left)
                else:
#                     print('at %d (%s) freq is greater than %f' % (N.idx, self.vocab[N.idx], N.value))
                    return classify(case, N.right)
                
        LEAF_SIZE = 100 # maximum points at a leaf
        SPLIT = 50
        DICT_SIZE = 500
        
        train_ham, train_spam = self.split_ham_spam(self.train_X, self.train_label)
        hmean, smean = train_ham.mean(axis = 0), train_spam.mean(axis = 0)

        freq_diff = np.absolute(hmean - smean) # difference of word freq in HAM vs. SPAM for each word
        arg_fdiff = np.flip(np.argsort(freq_diff)) # idx of word in descending order of freq diff
        arg_list = List(arg_fdiff[:DICT_SIZE])
        # use all data points(rows) to build the tree
        root = build_tree(np.arange(self.N_TRAIN.sum()))
        self.result = np.empty(self.N_TEST.sum())
        print('test...')
        for i in np.arange(np.size(self.test_label)):
            self.result[i] = classify(self.test_X[i], root)
            
        print('test took %.2f s' % (time.time() - start_time))
        return self.result