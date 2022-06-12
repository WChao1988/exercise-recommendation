# exercise-recommendation
# usage
# initail model
r2 = RandL2Embed(student_num, concept_num, target_space_n, lr=0.01)
# student number 
# concept number
# target space dimension
# lr learning rate
r2.train_now(train=train)
# train list of triples [[s1, c1, w11], [s1, c2, w12], [s2, c2, w22], [s3, c3, w33], .....], s1 the id of student, c1 the id of concepts
# w12 the weight between student 1 and concept 2
# test list of triples
r2.test_now(test=train)  
