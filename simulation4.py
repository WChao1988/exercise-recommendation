
from sbase import SvdEmbed, RandL1Embed, RandL2Embed, OneHotL1Embed, OneHotL2Embed, FeatureL1Embed, FeatureL2Embed, SimBase
from show import show_graph, my_plt, two_plt, plt_s_c
import numpy as np
import torch

"""3 students succeed in 3 concepts, the rating 
r_0,0 = r_1,0 = r_1,1 = r_2,0 = r_2,1 = r_2,2 = 1

"""


class Sim5(SimBase):
    def __init__(self):
        super(Sim5, self).__init__()
        self.student_num = 3
        self.concept_num = 3
        # full graph
        self.ge_all = self.full_graph()

    def full_graph(self):
        """return full graph: list of triples [student_id, concept_id, relation]
        relation = 0: failure   relation=1: success
        """
        graph = []
        for i in range(self.student_num):
            for j in range(self.concept_num):
                if i <= j:
                    relation = 1
                else:
                    relation = 0
                graph.append([i, j, relation])
        return graph


if __name__ == "__main__":
    generator_sm = Sim5()
    train, test = generator_sm.get_train_test(fky=1)
    student_num = 3
    concept_num = 3
    target_space_n = 3
    # # 测试0
    # s = SvdEmbed(student_num, concept_num, target_space_n, lr=0.01)
    # s.train_now(train=train)
    # s.test_now(test)
    # # 测试2
    # r2 = RandL2Embed(student_num, concept_num, target_space_n, lr=0.01)
    # r2.train_now(train=train)
    # r2.test_now(test=train)  # test= 0.003
    # show_graph(r2, generator_sm, test)
    # 测试1
    # r2 = RandL1Embed(student_num, concept_num, target_space_n, lr=0.01)
    # r2.train_now(train=train)
    # r1.test_now(test=train)  # test= 0.0003
    # show_graph(r1, generator_sm, test)
    # # 测试3
    # o1 = OneHotL1Embed(student_num, concept_num, target_space_n, lr=0.01)
    # o1.train_now(train=train)
    # o1.test_now(test=test)  # test= 2.889e-6
    # show_graph(o1, generator_sm, test)
    # # 测试4
    # o2 = OneHotL2Embed(student_num, concept_num, target_space_n, lr=0.01)
    # o2.train_now(train)
    # o2.test_now(test)  # test= 0.000108
    # show_graph(o2, generator_sm, test)
    # # 测试5
    f1 = FeatureL1Embed(student_num, concept_num, target_space_n, lr=0.001)
    f1.train_now(train)
    student = torch.Tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0.5, 0.5, 0.5]])
    concept = torch.Tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0.5, 0.5, 0.5]])
    ss = f1.embedding_student(student) + torch.Tensor([0.5])
    cc = f1.embedding_concept(concept) + torch.Tensor([0.5])
    plt_s_c(ss, cc)
    # f1.test_now(test)  # test= 0.00283
    # show_graph(f1, generator_sm, test)
    # # 测试6
    # f2 = FeatureL2Embed(student_num, concept_num, target_space_n, lr=0.01)
    # f2.train_now(train)
    # f2.test_now(test)  # test = 0.00147
    # show_graph(f2, generator_sm, test)
    # #
    # my_plt([r1, r2], train=train)




