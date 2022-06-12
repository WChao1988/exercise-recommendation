from sbase import RandL1Embed, RandL2Embed, OneHotL1Embed, OneHotL2Embed, FeatureL1Embed, FeatureL2Embed, SimBase
from show import show_graph, my_plt
import numpy as np


"""Weighted graph, there are 20 students and 20 concepts, the relation
between the i-th student and the j-th concept is calculated by
max{sin(i), sin(j), 0}, a graph with rich penetrations complete"""


class Sim3(SimBase):
    def __init__(self):
        super(Sim3, self).__init__()
        self.student_num = 20
        self.concept_num = 20
        # 全图
        self.ge_all = self.full_graph()

    def full_graph(self):
        ge_all = []
        for i in range(self.student_num):
            for j in range(self.concept_num):
                relation = max(np.sin(i), np.sin(j), 0)
                ge_all.append([i, j, relation])
        return ge_all


if __name__ == "__main__":
    generator_sm = Sim3()
    train, test = generator_sm.get_train_test(fky=1)
    student_num = 20
    concept_num = 20
    target_space_n = 3
    # 测试1
    r1 = RandL1Embed(student_num, concept_num, target_space_n, lr=0.05)
    r1.train_now(train=train)
    r1.test_now(test=train)  # acc=0.97 mse=0.019
    # show_graph(r1, generator_sm, test)
    # 测试2
    r2 = RandL2Embed(student_num, concept_num, target_space_n, lr=0.01)
    r2.train_now(train=train)
    r2.test_now(test=train)  # acc=0.9875 mse=0.019
    # show_graph(r2, generator_sm, test)

    # 绘图
    my_plt([r1, r2], train=train)

