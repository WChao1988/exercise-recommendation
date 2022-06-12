import torch
import torch.optim as optim
import torch.nn as nn
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import random


torch.manual_seed(3000)
random.seed(1234)


# 模型母版
class Embed(nn.Module, metaclass=ABCMeta):
    def __init__(self, student_num, concept_num, target_space_n, lr):
        super(Embed, self).__init__()
        self.student_num = student_num
        self.concept_num = concept_num
        # 编码层输入数量
        self.student_input_num = None
        self.concept_input_num = None
        # 输出层维度
        self.student_embedded_num = target_space_n
        self.concept_embedded_num = target_space_n
        self.lr = lr
        # 矩阵图
        self.matrix_graph = None

    @abstractmethod
    def init_matrix_graph(self, ge):
        """ge list of triple [student_id, concept_id, relation],
        initialize self.matrix_graph: self.student_num * self.concept_num"""
        pass

    @abstractmethod
    def encoding_student(self, ge):
        """ge list of triple [student_id, concept_id, relation],
        output the encoding of student, a tensor N * self.student_input_num"""
        pass

    @abstractmethod
    def encoding_concept(self, ge):
        """ge list of triple [student_id, concept_id, relation]
        output the encoding of concept, a tensor N * self.concept_input_num"""
        pass

    def form_tensor_graph(self, ge):
        """ge list of triple [student_id, concept_id, relation]
        output the tensor graph: N * (self.concept_input_num + self.student_intput_num + 1)"""
        encoded_student = self.encoding_student(ge)
        encoded_concept = self.encoding_concept(ge)
        relation = torch.Tensor([[item[2]] for item in ge])
        return torch.cat((encoded_student, encoded_concept, relation), dim=1)

    @abstractmethod
    def embedding_student(self, encoded_student):
        """encoded_student: N * self.student_input_num as input
        output: the embedded vector of students, N * self.student_embedded_num"""
        pass

    @abstractmethod
    def embedding_concept(self, encoded_concept):
        """encoded_student: N * self.concept_input_num as input
        output: the embedded vector of concepts, N * self.concept_embedded_num"""
        pass

    @abstractmethod
    def metric(self, embedded_student, embedded_concept):
        """embedded_student:N * self.student_embedded_num;
        embedded_concept: N * self.concept_embedded_num,
        output: the distance between students with concepts, N Tensor"""
        pass

    def forward(self, input_tensor_graph):
        """input_tensor_graph: N * (self.concept_input_num + self.student_intput_num + 1)
        output: N Tensor"""
        encoded_student = input_tensor_graph[:, :self.student_input_num]
        encoded_concept = input_tensor_graph[:, self.student_input_num:-1]
        embedded_student = self.embedding_student(encoded_student)
        embedded_concept = self.embedding_concept(encoded_concept)
        return self.metric(embedded_student, embedded_concept)

    def train_now(self, train, nound=2500):
        self.init_matrix_graph(train)
        # 获取训练图
        tensor_graph = self.form_tensor_graph(train)
        target = tensor_graph[:, -1]
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        plt.ion()
        lss_to_plt = []
        for _ in range(nound):
            output = self.forward(tensor_graph)
            loss = criterion(output, target)
            print(loss)
            # print(student_bs1[8:9, :8])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lss_to_plt.append(loss.item())
            plt.clf()
            plt.plot(lss_to_plt[10:])
            plt.pause(0.00001)
            plt.ioff()

    def test_now(self, test):
        # 获取测试图
        tensor_graph = self.form_tensor_graph(test)
        target = tensor_graph[:, -1]
        output = self.forward(tensor_graph)
        loss = torch.nn.MSELoss()(output, target).item()
        # 准确率
        acc = 0
        for i, op in enumerate(output):
            if op > 0.51 and target[i] > 0.51:
                acc += 1
            elif op < 0.49 and target[i] < 0.49:
                acc += 1
        acc = acc / len(output)
        return acc, loss, output.data

    def matrix_predict_graph(self, test):
        tensor_graph = self.form_tensor_graph(test)
        output = self.forward(tensor_graph)
        mat_predict_graph = np.ones((self.student_num, self.concept_num)) / 2
        for i, ot in enumerate(output):
            mat_predict_graph[test[i][0], test[i][1]] = ot.item()
        return mat_predict_graph


class RandEmbed(Embed, metaclass=ABCMeta):
        def __init__(self, student_num, concept_num, target_space_n, lr):
            super(RandEmbed, self).__init__(student_num, concept_num, target_space_n, lr)
            self.student_input_num = student_num
            self.concept_input_num = concept_num
            # embedding layer student
            self.student_layer0 = nn.Linear(self.student_input_num, self.student_embedded_num)
            # embedding layer concept
            self.concept_layer0 = nn.Linear(self.concept_input_num, self.concept_embedded_num)

        def init_matrix_graph(self, ge):
            """ge list of triple [student_id, concept_id, relation],
            initialize self.matrix_graph: self.student_num * self.concept_num"""
            self.matrix_graph = np.ones((self.student_num, self.concept_num)) / 2
            for gei in ge:
                self.matrix_graph[gei[0], gei[1]] = gei[2]

        def encoding_student(self, ge):
            """ge list of triple [student_id, concept_id, relation],
            output the encoding of student, a tensor N * self.student_input_num"""
            n = len(ge)
            encoded_student = torch.zeros(n, self.student_num)
            for i, item in enumerate(ge):
                encoded_student[i, item[0]] = 1
            return encoded_student

        def encoding_concept(self, ge):
            """ge list of triple [student_id, concept_id, relation]
            output the encoding of concept, a tensor N * self.concept_input_num"""
            n = len(ge)
            encoded_concept = torch.zeros(n, self.concept_num)
            for i, item in enumerate(ge):
                encoded_concept[i, item[1]] = 1
            return encoded_concept

        def embedding_student(self, encoded_student):
            """encoded_student: N * self.student_input_num as input
            output: the embedded vector of students, N * self.student_embedded_num"""
            embedded_student = self.student_layer0(encoded_student)
            return embedded_student

        def embedding_concept(self, encoded_concept):
            """encoded_student: N * self.concept_input_num as input
            output: the embedded vector of concepts, N * self.concept_embedded_num"""
            embedded_concept = self.concept_layer0(encoded_concept)
            return embedded_concept

        @abstractmethod
        def metric(self, embedded_student, embedded_concept):
            """embedded_student:N * self.student_embedded_num;
            embedded_concept: N * self.concept_embedded_num,
            output: the distance between students with concepts, N Tensor"""
            pass


class OneHotEmbed(Embed, metaclass=ABCMeta):
    def __init__(self, student_num, concept_num, target_space_n, lr):
        super(OneHotEmbed, self).__init__(student_num, concept_num, target_space_n, lr)
        self.student_input_num = student_num
        self.concept_input_num = concept_num
        # embedding layer student
        self.student_layer0 = nn.Linear(self.student_input_num, self.student_embedded_num)
        # self.student_layer1 = nn.Linear(512, 512)
        # self.student_layer2 = nn.Linear(512, 512)
        # self.student_layer3 = nn.Linear(512, 512)
        # self.student_layer4 = nn.Linear(32, self.student_embedded_num)
        # embedding layer concept
        self.concept_layer0 = nn.Linear(self.concept_input_num, self.concept_embedded_num)
        # self.concept_layer1 = nn.Linear(512, 512)
        # self.concept_layer2 = nn.Linear(512, 512)
        # self.concept_layer3 = nn.Linear(512, 512)
        # self.concept_layer4 = nn.Linear(32, self.concept_embedded_num)

    def init_matrix_graph(self, ge):
        """ge list of triple [student_id, concept_id, relation],
        initialize self.matrix_graph: self.student_num * self.concept_num"""
        self.matrix_graph = np.ones((self.student_num, self.concept_num)) / 2
        for gei in ge:
            self.matrix_graph[gei[0], gei[1]] = gei[2]

    def encoding_student(self, ge):
        """ge list of triple [student_id, concept_id, relation],
        output the encoding of student, a tensor N * self.student_input_num"""
        n = len(ge)
        encoded_student = torch.zeros(n, self.student_num)
        for i, item in enumerate(ge):
            encoded_student[i, item[0]] = 1
        return encoded_student

    def encoding_concept(self, ge):
        """ge list of triple [student_id, concept_id, relation]
        output the encoding of concept, a tensor N * self.concept_input_num"""
        n = len(ge)
        encoded_concept = torch.zeros(n, self.concept_num)
        for i, item in enumerate(ge):
            encoded_concept[i, item[1]] = 1
        return encoded_concept

    def embedding_student(self, encoded_student):
        """encoded_student: N * self.student_input_num as input
        output: the embedded vector of students, N * self.student_embedded_num"""
        embedded_student = self.student_layer0(encoded_student)
        # embedded_student = self.student_layer1(embedded_student)
        # embedded_student = self.student_layer2(embedded_student)
        # embedded_student = self.student_layer3(embedded_student)
        # embedded_student = self.student_layer4(embedded_student)
        return embedded_student

    def embedding_concept(self, encoded_concept):
        """encoded_student: N * self.concept_input_num as input
        output: the embedded vector of concepts, N * self.concept_embedded_num"""
        embedded_concept = self.concept_layer0(encoded_concept)
        # embedded_concept = self.concept_layer1(embedded_concept)
        # embedded_concept = self.concept_layer2(embedded_concept)
        # embedded_concept = self.concept_layer3(embedded_concept)
        # embedded_concept = self.concept_layer4(embedded_concept)
        return embedded_concept

    @abstractmethod
    def metric(self, embedded_student, embedded_concept):
        """embedded_student:N * self.student_embedded_num;
        embedded_concept: N * self.concept_embedded_num,
        output: the distance between students with concepts, N Tensor"""
        pass


class FeatureEmbed(Embed, metaclass=ABCMeta):
    def __init__(self, student_num, concept_num, target_space_n, lr):
        super(FeatureEmbed, self).__init__(student_num, concept_num, target_space_n, lr)
        self.student_input_num = concept_num
        self.concept_input_num = student_num
        # embedding layer student
        self.student_layer0 = nn.Linear(self.student_input_num, self.student_embedded_num)
        # self.student_layer1 = nn.Linear(512, 512)
        # self.student_layer2 = nn.Linear(512, 512)
        # self.student_layer3 = nn.Linear(512, 512)
        # self.student_layer4 = nn.Linear(64, self.student_embedded_num)
        # embedding layer concept
        self.concept_layer0 = nn.Linear(self.concept_input_num, self.concept_embedded_num)
        # self.concept_layer1 = nn.Linear(512, 512)
        # self.concept_layer2 = nn.Linear(512, 512)
        # self.concept_layer3 = nn.Linear(512, 512)
        # self.concept_layer4 = nn.Linear(64, self.concept_embedded_num)

    def init_matrix_graph(self, ge):
        """ge list of triple [student_id, concept_id, relation],
        initialize self.matrix_graph: self.student_num * self.concept_num"""
        self.matrix_graph = np.ones((self.student_num, self.concept_num)) / 2
        for gei in ge:
            self.matrix_graph[gei[0], gei[1]] = gei[2]

    def encoding_student(self, ge):
        """ge list of triple [student_id, concept_id, relation],
        output the encoding of student, a tensor N * self.student_input_num"""
        # 构建特征编码
        n = len(ge)
        encoded_student = torch.zeros(n, self.student_input_num)
        for i, gei in enumerate(ge):
            encoded_student[i, :] = torch.from_numpy(self.matrix_graph[gei[0], :])
        return encoded_student

    def encoding_concept(self, ge):
        """ge list of triple [student_id, concept_id, relation]
        output the encoding of concept, a tensor N * self.concept_input_num"""
        # 构建特征编码
        n = len(ge)
        encoded_concept = torch.zeros(n, self.concept_input_num)
        for i, gei in enumerate(ge):
            encoded_concept[i, :] = torch.from_numpy(self.matrix_graph[:, gei[1]])
        return encoded_concept

    def embedding_student(self, encoded_student):
        """encoded_student: N * self.student_input_num as input
        output: the embedded vector of students, N * self.student_embedded_num"""
        embedded_student = self.student_layer0(encoded_student)
        # embedded_student = self.student_layer1(embedded_student)
        # embedded_student = self.student_layer2(embedded_student)
        # embedded_student = self.student_layer3(embedded_student)
        # embedded_student = self.student_layer4(embedded_student)
        return embedded_student

    def embedding_concept(self, encoded_concept):
        """encoded_student: N * self.concept_input_num as input
        output: the embedded vector of concepts, N * self.concept_embedded_num"""
        embedded_concept = self.concept_layer0(encoded_concept)
        # embedded_concept = self.concept_layer1(embedded_concept)
        # embedded_concept = self.concept_layer2(embedded_concept)
        # embedded_concept = self.concept_layer3(embedded_concept)
        # embedded_concept = self.concept_layer4(embedded_concept)
        return embedded_concept

    @abstractmethod
    def metric(self, embedded_student, embedded_concept):
        """embedded_student:N * self.student_embedded_num;
        embedded_concept: N * self.concept_embedded_num,
        output: the distance between students with concepts, N Tensor"""
        pass


class SvdEmbed(OneHotEmbed):
    def __init__(self, student_num, concept_num, target_space_n, lr=0.01):
        super(SvdEmbed, self).__init__(student_num, concept_num, target_space_n, lr)

    def form_tensor_graph(self, ge):
        """ge list of triple [student_id, concept_id, relation]
        output the tensor graph: N * (self.concept_input_num + self.student_intput_num + 1)"""
        encoded_student = self.encoding_student(ge)
        encoded_concept = self.encoding_concept(ge)
        relation = torch.Tensor([[item[2]] for item in ge])
        return torch.cat((encoded_student, encoded_concept, 2 * relation - 1), dim=1)

    def test_now(self, test):
        # 获取测试图
        tensor_graph = self.form_tensor_graph(test)
        target = tensor_graph[:, -1]
        output = self.forward(tensor_graph)
        loss = torch.nn.MSELoss()(output, target).item()
        # 准确率
        acc = 0
        for i, op in enumerate(output):
            if op > 0.01 and target[i] > 0.01:
                acc += 1
            elif op < -0.01 and target[i] < -0.01:
                acc += 1
        acc = acc / len(output)
        return acc, loss, output.data

    def metric(self, embedded_student, embedded_concept):
        coordinate_diff = embedded_student * embedded_concept
        distance = torch.sum(coordinate_diff, dim=1)
        return distance

    def train_now(self, train, nound=2500):
        self.init_matrix_graph(train)
        # 获取训练图
        tensor_graph = self.form_tensor_graph(train)
        target = tensor_graph[:, -1]
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        plt.ion()
        lss_to_plt = []
        for _ in range(nound):
            output = self.forward(tensor_graph)
            loss = criterion(output, target)
            print(loss)
            # print(student_bs1[8:9, :8])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lss_to_plt.append(loss.item())
            plt.clf()
            plt.plot(lss_to_plt[10:])
            plt.pause(0.00001)
            plt.ioff()

class BiasSvdEmbed(RandEmbed):
    def __init__(self, student_num, concept_num, target_space_n, lr=0.01):
        super(BiasSvdEmbed, self).__init__(student_num, concept_num, target_space_n, lr)

    def form_tensor_graph(self, ge):
        """ge list of triple [student_id, concept_id, relation]
        output the tensor graph: N * (self.concept_input_num + self.student_intput_num + 1)"""
        encoded_student = self.encoding_student(ge)
        encoded_concept = self.encoding_concept(ge)
        relation = torch.Tensor([[item[2]] for item in ge])
        return torch.cat((encoded_student, encoded_concept, 2 * relation - 1), dim=1)

    def test_now(self, test):
        # 获取测试图
        tensor_graph = self.form_tensor_graph(test)
        target = tensor_graph[:, -1]
        output = self.forward(tensor_graph)
        loss = torch.nn.MSELoss()(output, target).item()
        # 准确率
        acc = 0
        for i, op in enumerate(output):
            if op > 0.01 and target[i] > 0.01:
                acc += 1
            elif op < -0.01 and target[i] < -0.01:
                acc += 1
        acc = acc / len(output)
        return acc, loss, output.data

    def metric(self, embedded_student, embedded_concept):
        inner_dot = embedded_student[:, 1:] * embedded_concept[:, 1:]
        distance = torch.sum(inner_dot, dim=1) + embedded_student[:, 0] + embedded_concept[:, 0]
        return distance


class RandL2Embed(RandEmbed):
    def __init__(self, student_num, concept_num, target_space_n, lr=0.01):
        super(RandL2Embed, self).__init__(student_num, concept_num, target_space_n, lr)

    def metric(self, embedded_student, embedded_concept):
        coordinate_diff = embedded_student - embedded_concept
        distance = torch.norm(coordinate_diff, dim=1, p=2)
        return distance


class RandL1Embed(RandEmbed):
    def __init__(self, student_num, concept_num, target_space_n, lr=0.01):
        super(RandL1Embed, self).__init__(student_num, concept_num, target_space_n, lr)

    def metric(self, embedded_student, embedded_concept):
        coordinate_diff = nn.ReLU()(embedded_student - embedded_concept)
        distance = torch.norm(coordinate_diff, dim=1, p=1)
        return distance


class OneHotL2Embed(OneHotEmbed):
    def __init__(self, student_num, concept_num, target_space_n, lr=0.01):
        super(OneHotL2Embed, self).__init__(student_num, concept_num, target_space_n, lr)

    def metric(self, embedded_student, embedded_concept):
        coordinate_diff = embedded_student - embedded_concept
        distance = torch.norm(coordinate_diff, dim=1, p=2)
        return distance


class OneHotL1Embed(OneHotEmbed):
    def __init__(self, student_num, concept_num, target_space_n, lr=0.01):
        super(OneHotL1Embed, self).__init__(student_num, concept_num, target_space_n, lr)

    def metric(self, embedded_student, embedded_concept):
        coordinate_diff = nn.ReLU()(embedded_student - embedded_concept)
        distance = torch.norm(coordinate_diff, dim=1, p=1)
        return distance


class FeatureL2Embed(FeatureEmbed):
    def __init__(self, student_num, concept_num, target_space_n, lr=0.01):
        super(FeatureL2Embed, self).__init__(student_num, concept_num, target_space_n, lr)

    def metric(self, embedded_student, embedded_concept):
        coordinate_diff = embedded_student - embedded_concept
        distance = torch.norm(coordinate_diff, dim=1, p=2)
        return distance


class FeatureL1Embed(FeatureEmbed):
    def __init__(self, student_num, concept_num, target_space_n, lr=0.001):
        super(FeatureL1Embed, self).__init__(student_num, concept_num, target_space_n, lr)

    def form_tensor_graph(self, ge):
        """ge list of triple [student_id, concept_id, relation]
        output the tensor graph: N * (self.concept_input_num + self.student_intput_num + 1)"""
        encoded_student = self.encoding_student(ge)
        encoded_concept = self.encoding_concept(ge)
        relation = torch.Tensor([[item[2]] for item in ge])
        return torch.cat((encoded_student, encoded_concept, 1 - relation), dim=1)

    def train_now(self, train, nound=2500):
        self.init_matrix_graph(train)
        # 获取训练图
        tensor_graph = self.form_tensor_graph(train)
        target = tensor_graph[:, -1]
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        plt.ion()
        lss_to_plt = []
        for _ in range(nound):
            output = self.forward(tensor_graph)
            loss = criterion(output, target)
            print(loss)
            # print(student_bs1[8:9, :8])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lss_to_plt.append(loss.item())
            plt.clf()
            plt.plot(lss_to_plt[10:])
            plt.pause(0.00001)
            plt.ioff()

    def test_now(self, test):
        # 获取测试图
        tensor_graph = self.form_tensor_graph(test)
        target = tensor_graph[:, -1]
        output = self.forward(tensor_graph)
        loss = torch.nn.MSELoss()(output, target).item()
        # 准确率
        acc = 0
        for i, op in enumerate(output):
            if op > 0.5 and target[i] > 0.50001:
                acc += 1
            elif op < 0.5 and target[i] < 0.49999:
                acc += 1
        acc = acc / len(output)
        return acc, loss, output.data

    def metric(self, embedded_student, embedded_concept):
        coordinate_diff = nn.ReLU()(embedded_student - embedded_concept)
        # coordinate_diff = embedded_concept - embedded_student
        distance = torch.norm(coordinate_diff, dim=1, p=1)
        return distance


class SimBase(metaclass=ABCMeta):
    def __init__(self):
        # 学生数，概念数
        self.student_num = None
        self.concept_num = None
        # 完全图或者训练图
        self.ge_all = None

    @abstractmethod
    def full_graph(self):
        """return full graph: list of triples [student_id, concept_id, relation]
        relation = 0: failure   relation=1: success
        """
        pass

    def get_train_test(self, fky=0.8):
        """return two lists: train and test, both filled with triples"""
        train_now = []
        test_now = []
        for gei in self.ge_all:
            if random.uniform(0, 1) < fky:
                train_now.append(gei)
            else:
                test_now.append(gei)
        return train_now, test_now

    def matrix_graph(self, ge):
        """ge a graph list: list of triples [student_id, concept_id, relation]
        return an array: self.student_num * self.concept_num;
        usually: filled the nan elements with 0.5"""
        mat = np.ones((self.student_num, self.concept_num)) / 2
        for gei in ge:
            mat[gei[0], gei[1]] = gei[2]
        return mat


class UserFilter:
    def __init__(self, student_num, concept_num, first_k=3):
        self.student_num = student_num
        self.concept_num = concept_num
        self.rating_matrix = None
        self.relation_matrix = None
        self.pre_matrix = None
        self.k = first_k

    def __init_rating_matrix__(self, train_graph):
        self.rating_matrix = np.ones((self.student_num, self.concept_num))/2
        for gr in train_graph:
            self.rating_matrix[gr[0], gr[1]] = gr[2]

    def __init_relation_matrix__(self):
        self.relation_matrix = np.zeros((self.student_num, self.student_num))
        for i in range(self.student_num):
            for j in range(i + 1, self.student_num):
                ui = self.rating_matrix[i, :]
                uj = self.rating_matrix[j, :]
                norm_ui = np.linalg.norm(ui, 2)
                norm_uj = np.linalg.norm(uj, 2)
                self.relation_matrix[i, j] = np.dot(ui, uj) / norm_ui / norm_uj
                # self.relation_matrix[i, j] = 1 - np.linalg.norm(ui - uj, 2)
        self.relation_matrix = self.relation_matrix + np.transpose(self.relation_matrix)
        for i in range(self.student_num):
            self.relation_matrix[i, :] = self.my_mute(self.relation_matrix[i, :], self.k)
        for i in range(self.student_num):
            self.relation_matrix[i, :] = self.relation_matrix[i, :] / np.sum(self.relation_matrix[i, :])


    @classmethod
    def my_mute(cls, x, k):
        q = np.sort(x)[-(k -1)]
        x[x < q] = 0
        return x

    def train_now(self, train):
        """input: triplet [i, j , r]"""
        self.__init_rating_matrix__(train_graph=train)
        self.__init_relation_matrix__()
        self.pre_matrix = np.matmul(self.relation_matrix, self.rating_matrix)

    def test_now(self, test):
        acc = 0
        mse = 0
        pre = np.zeros(len(test))
        for i, ti in enumerate(test):
            pre[i] = self.pre_matrix[ti[0], ti[1]]
            if pre[i] > 0.5 and ti[2] > 0.5:
                acc += 1
            elif pre[i] < 0.5 and ti[2] < 0.5:
                acc += 1
            mse = (pre[i] - ti[2])**2
        return acc/len(test), mse/len(test), pre

    def _get_name(self):
        return 'UserFilter'

class ItemFilter:
    def __init__(self, student_num, concept_num, first_k=15):
        self.student_num = student_num
        self.concept_num = concept_num
        self.rating_matrix = None
        self.relation_matrix = None
        self.pre_matrix = None
        self.k = first_k

    def __init_rating_matrix__(self, train_graph):
        self.rating_matrix = np.ones((self.student_num, self.concept_num))/2
        for gr in train_graph:
            self.rating_matrix[gr[0], gr[1]] = gr[2]

    def __init_relation_matrix__(self):
        self.relation_matrix = np.zeros((self.concept_num, self.concept_num))
        for i in range(self.concept_num):
            for j in range(i + 1, self.concept_num):
                ii = self.rating_matrix[:, i]
                ij = self.rating_matrix[:, j]
                norm_ii = np.linalg.norm(ii, 2)
                norm_ij = np.linalg.norm(ij, 2)
                self.relation_matrix[i, j] = np.dot(ii, ij) / norm_ii / norm_ij
        self.relation_matrix = self.relation_matrix + np.transpose(self.relation_matrix)
        for i in range(self.concept_num):
            self.relation_matrix[:, i] = self.my_mute(self.relation_matrix[:, i], self.k)
        for i in range(self.concept_num):
            self.relation_matrix[:, i] = self.relation_matrix[:, i] / np.sum(self.relation_matrix[:, i])

    @classmethod
    def my_mute(cls, x, k):
        q = np.sort(x)[-k]
        x[x < q] = 0
        return x

    def train_now(self, train):
        """input: triplet [i, j , r]"""
        self.__init_rating_matrix__(train_graph=train)
        self.__init_relation_matrix__()
        self.pre_matrix = np.matmul(self.rating_matrix, self.relation_matrix)

    def test_now(self, test):
        acc = 0
        mse = 0
        pre = np.zeros(len(test))
        for i, ti in enumerate(test):
            pre[i] = self.pre_matrix[ti[0], ti[1]]
            if pre[i] > 0.5001 and ti[2] > 0.5001:
                acc += 1
            elif pre[i] < 0.4999 and ti[2] < 0.4999:
                acc += 1
            mse = (pre[i] - ti[2])**2
        return acc/len(test), mse/len(test), pre

    def _get_name(self):
        return 'ItemFilter'


class lstmKC(nn.Module):
    def __init__(self, student_num, concept_num, target_space=8, lr=0.01):
        super(lstmKC, self).__init__()
        self.student_num = student_num
        self.concept_num = concept_num
        self.lr = lr
        self.lstm_stu = nn.LSTM(self.student_num, target_space)
        self.lstm_con = nn.LSTM(self.concept_num, target_space)

    def form_input(self, train):
        n = len(train)
        student_input = torch.zeros(n, self.student_num)
        concept_input = torch.zeros(n, self.concept_num)
        target = torch.zeros(n)
        for i, tri in enumerate(train):
            student_input[i, tri[0]] = 1
            concept_input[i, tri[1]] = 1
            target[i] = tri[2]
        return student_input, concept_input, target

    def forward(self, student_input, concept_input):
        student_embed, _ = self.lstm_stu(student_input)
        concept_embed, _ = self.lstm_con(concept_input)
        inner_product = student_embed * concept_embed
        inner_product1 = torch.sum(inner_product, dim=1)
        return inner_product1

    def train_now(self, train, nround=1000):
        student_input, concept_input, target = self.form_input(train)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        lss = []
        plt.ion()
        for i in range(nround):
            output = self.forward(student_input, concept_input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lss.append(loss.item())
            plt.clf()
            plt.plot(lss[10:])
            plt.pause(0.00001)
            plt.ioff()
            print(loss.item())

    def test_now(self, test):
        student_input, concept_input, target = self.form_input(test)
        output = self.forward(student_input, concept_input)
        acc = 0
        mse = 0
        for i, ti in enumerate(test):
            value = output[i].item()
            if value > 0.5001 and ti[2] > 0.5001:
                acc += 1
            elif value < 0.4999 and ti[2] < 0.4999:
                acc += 1
            mse = (value - ti[2]) ** 2
        return acc / len(test), mse / len(test), output


class TopK:
    def __init__(self, model, test, student_num):
        self.model = model
        self.test = test
        self.student_num = student_num
        acc, mse, self.prediction = self.model.test_now(test=test)

    def little_top_k(self, stu_id, k):
        stu = [[item, self.prediction[i]] for i, item in enumerate(self.test) if item[0] == stu_id]
        if len(stu) == 0:
            return 0.5
        stu_top = sorted(stu, key=lambda item: item[1], reverse=False)[:k]
        num = 0
        for item in stu_top:
            if item[0][2] < 0.5:
                num += 1
        return num / len(stu_top)

    def big_top_k(self, stu_id, k):
        stu = [[item, self.prediction[i]] for i, item in enumerate(self.test) if item[0] == stu_id]
        if len(stu) == 0:
            return 0.5
        stu_top = sorted(stu, key=lambda item: item[1], reverse=True)[:k]
        num = 0
        for item in stu_top:
            if item[0][2] < 0.5:
                num += 1
        return num / len(stu_top)

    def top_k(self, k):
        if self.model._get_name() == 'FeatureL1Embed':
            chosen_top_k = self.big_top_k
        else:
            chosen_top_k = self.little_top_k
        top_acc = 0
        for i in range(self.student_num):
            top_acc = top_acc + chosen_top_k(i, k)
        return top_acc / self.student_num

