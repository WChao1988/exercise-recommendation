from sbase import RandL1Embed, RandL2Embed, FeatureL1Embed, SvdEmbed, BiasSvdEmbed, UserFilter, ItemFilter, lstmKC, TopK
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn
import pickle


class Algebra:
    def __init__(self):
        self.graph = None
        self.__form__graph__()
        self.student_num = None
        self.concept_num = None
        self.student_dict = {}
        self.concept_dict = {}
        self.all_graph_num = self.get_all_graph()
        self.rating_matrix = None
        self.positive_index = []
        self.negative_index = []
        self.__form__rating_matrix__()
        # self.train_graph, self.test_graph = self.form_train_test()
        self.train_rating_matrix = None
        self.test_rating_matrix = None
        self.train_graph, self.test_graph = self.form_train_test2()

    def __form__graph__(self):
        f = open('./mathlog/ds1275_student_step_All_Data_2872_2015_0620_120737.txt', 'r')
        one_line = f.readlines(1)
        one_line = f.readlines(1)
        graph = []
        correct_dict = {'correct': 1, 'incorrect': -1, 'hint': 0}
        while one_line:
            one_line_list = one_line[0].split('\t')
            student = one_line_list[2]
            concept = one_line_list[31]
            correct = one_line_list[14]
            graph.append([student, concept, correct_dict[correct]])
            one_line = f.readlines(1)
        self.graph = graph
        return graph

    def get_all_graph(self):
        students = [item[0] for item in self.graph]
        concepts = [item[1] for item in self.graph if item[1] != '']
        self.student_num = len(set(students))
        self.concept_num = len(set(concepts))
        for i, student in enumerate(set(students)):
            self.student_dict[student] = i
        for j, concept in enumerate(set(concepts)):
            self.concept_dict[concept] = j
        all_graph = [[self.student_dict[item[0]], self.concept_dict[item[1]], item[2]] for item in self.graph if item[1] != '']
        return all_graph

    def __form__rating_matrix__(self):
        self.rating_matrix = np.zeros((self.student_num, self.concept_num))
        for gr in self.all_graph_num:
            self.rating_matrix[gr[0], gr[1]] = self.rating_matrix[gr[0], gr[1]] + gr[2]

        for i in range(self.student_num):
            for j in range(self.concept_num):
                if self.rating_matrix[i, j] > 0.1:
                    self.positive_index.append([i, j])
                elif self.rating_matrix[i, j] < -0.1:
                    self.negative_index.append([i, j])
                self.rating_matrix[i, j] = 1 / (1 + np.exp(-self.rating_matrix[i, j]))

    def form_train_test(self):
        n = int(len(self.negative_index) * 0.3)
        train1, test1 = train_test_split(self.negative_index, test_size=n, random_state=3333)
        train2, test2 = train_test_split(self.positive_index, test_size=2 * n, random_state=4444)
        train = train1 + train2
        test = test1 + test2
        train_graph = []
        for tri in train:
            train_graph.append([tri[0], tri[1], self.rating_matrix[tri[0], tri[1]]])
        test_graph = []
        for tei in test:
            test_graph.append([tei[0], tei[1], self.rating_matrix[tei[0], tei[1]]])
        return train_graph, test_graph

    def form_train_test2(self):
        n = int(len(self.all_graph_num) * 0.2)
        train, test = train_test_split(self.all_graph_num, test_size=n)
        self.train_rating_matrix = np.zeros((self.student_num, self.concept_num))
        self.test_rating_matrix = np.zeros((self.student_num, self.concept_num))
        for tri in train:
            self.train_rating_matrix[tri[0], tri[1]] = self.train_rating_matrix[tri[0], tri[1]] + tri[2]
        for tei in test:
            self.test_rating_matrix[tei[0], tei[1]] = self.test_rating_matrix[tei[0], tei[1]] + tei[2]
        train_graph = []
        test_graph = []
        for i in range(self.student_num):
            for j in range(self.concept_num):
                train_rating_ij = self.train_rating_matrix[i, j]
                test_rating_ij = self.test_rating_matrix[i, j]
                if train_rating_ij != 0:
                    train_graph.append([i, j, 1 / (1 + np.exp(-train_rating_ij))])
                if test_rating_ij != 0:
                    test_graph.append([i, j, 1 / (1 + np.exp(-test_rating_ij))])
        return train_graph, test_graph


class Experiment2:
    def __init__(self, new_try=True):
        if new_try:
            gotry = Algebra()
            pickle.dump(gotry, open('gotry1.pkl', 'wb'))
        else:
            gotry = pickle.load(open('gotry1.pkl', 'rb'))
        self.gotry = gotry
        self.student_num = gotry.student_num
        self.concept_num = gotry.concept_num
        self.rating_matrix = gotry.rating_matrix
        self.train_graph = gotry.train_graph
        self.test_graph = gotry.test_graph
        # sensitive
        self.r1 = None
        self.r2 = None
        self.f1 = None
        self.s1 = None
        self.s2 = None
        self.u1 = None
        self.i1 = None
        self.l1 = None

    def train_by_sensitive(self):
        r1 = RandL1Embed(self.student_num, self.concept_num, target_space_n=13, lr=0.01)
        r1.train_now(train=self.train_graph)
        self.r1 = r1
        return r1

    def train_by_transe(self):
        r2 = RandL2Embed(self.student_num, self.concept_num, target_space_n=13, lr=0.01)
        r2.train_now(train=self.train_graph)
        self.r2 = r2
        return r2

    def train_by_feature_l1(self, target_space_n=13):
        f1 = FeatureL1Embed(self.student_num, self.concept_num, target_space_n=target_space_n, lr=0.001)
        f1.train_now(train=self.train_graph, nound=4500)
        self.f1 = f1
        return f1

    def train_by_svd(self):
        s1 = SvdEmbed(self.student_num, self.concept_num, target_space_n=13, lr=0.01)
        s1.train_now(train=self.train_graph)
        self.s1 = s1
        return s1

    def train_by_biasSvd(self):
        s2 = BiasSvdEmbed(self.student_num, self.concept_num, target_space_n=13, lr=0.01)
        s2.train_now(train=self.train_graph)
        self.s2 = s2
        return s2

    def train_by_ufilter(self, first_k):
        u1 = UserFilter(self.student_num, self.concept_num, first_k=first_k)
        u1.train_now(train=self.train_graph)
        self.u1 = u1
        return u1

    def train_by_ifilter(self, first_k):
        i1 = ItemFilter(self.student_num, self.concept_num, first_k=first_k)
        i1.train_now(train=self.train_graph)
        self.i1 = i1
        return i1

    def train_by_lstmKC(self):
        l1 = lstmKC(self.student_num, self.concept_num, target_space=13, lr=0.01)
        l1.train_now(train=self.train_graph, nround=500)
        self.l1 = l1
        return l1

exper2 = Experiment2(new_try=True)
# # random l1
r1 = exper2.train_by_sensitive()
# # # random l2
# r2 = exper2.train_by_transe()
# # feature l1
# #
f1 = exper2.train_by_feature_l1(13)
# # #
# s1 = exper2.train_by_svd()
# s2 = exper2.train_by_biasSvd()
# u1 = exper2.train_by_ufilter(first_k=13)
# i1 = exper2.train_by_ifilter(first_k=13)
l1 = exper2.train_by_lstmKC()

# acc_r1, mse_r1, prediction_r1 = r1.test_now(exper2.test_graph)
# acc_r2, mse_r2, prediction_r2 = r2.test_now(exper2.test_graph)
# acc_f1, mse_f1, prediction_f1 = f1.test_now(exper2.test_graph)
# acc_s1, mse_s1, prediction_s1 = s1.test_now(exper2.test_graph)
# acc_s2, mse_s2, prediction_s2 = s2.test_now(exper2.test_graph)
# acc_u1, mse_u1, prediction_u1 = u1.test_now(exper2.test_graph)
# acc_i1, mse_i1, prediction_i1 = i1.test_now(exper2.test_graph)
acc_l1, mse_l1, prediction_l1 = l1.test_now(exper2.test_graph)


# acc_r2_5 = TopK(model=r2, test=exper2.test_graph, student_num=exper2.student_num).top_k(5)
# acc_r2_10 = TopK(model=r2, test=exper2.test_graph, student_num=exper2.student_num).top_k(10)
acc_r1_5 = TopK(model=r1, test=exper2.test_graph, student_num=exper2.student_num).top_k(5)
acc_r1_10 = TopK(model=r1, test=exper2.test_graph, student_num=exper2.student_num).top_k(10)
# acc_u1_5 = TopK(model=u1, test=exper2.test_graph, student_num=exper2.student_num).top_k(5)
# acc_u1_10 = TopK(model=u1, test=exper2.test_graph, student_num=exper2.student_num).top_k(10)
# acc_i1_5 = TopK(model=i1, test=exper2.test_graph, student_num=exper2.student_num).top_k(5)
# acc_i1_10 = TopK(model=i1, test=exper2.test_graph, student_num=exper2.student_num).top_k(10)
# acc_s1_5 = TopK(model=s1, test=exper2.test_graph, student_num=exper2.student_num).top_k(5)
# acc_s1_10 = TopK(model=s1, test=exper2.test_graph, student_num=exper2.student_num).top_k(10)
# acc_s2_5 = TopK(model=s2, test=exper2.test_graph, student_num=exper2.student_num).top_k(5)
# acc_s2_10 = TopK(model=s2, test=exper2.test_graph, student_num=exper2.student_num).top_k(10)
acc_f1_5 = TopK(model=f1, test=exper2.test_graph, student_num=exper2.student_num).top_k(5)
acc_f1_10 = TopK(model=f1, test=exper2.test_graph, student_num=exper2.student_num).top_k(10)
acc_l1_5 = TopK(model=l1, test=exper2.test_graph, student_num=exper2.student_num).top_k(5)
acc_l1_10 = TopK(model=l1, test=exper2.test_graph, student_num=exper2.student_num).top_k(10)