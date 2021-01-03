# -*- coding: utf-8 -*-
"""
@author: yukti
"""


from LanguageModelling import NGramModel



# Getting Data/Corpus to use for N-Gram model
N = 3
N = int(input("Please enter the value of N for the N-Gram Model = "))
doc = open("doc1.txt","r")

data1 = "Ronald read her book. I read a different book. Jake read a book by Twain! "
test_sentence1 = "Ronald read a book"

data2 = doc.read()
test_sentence2 = "compare this increased efficiency of hours"

data3 = (open("doc2.txt","r")).read()
test_sentence3 = "people are much busier than their ancestors"



model_obj = NGramModel(N,data3)
model_obj.Language_Modelling(test_sentence3)
model_obj.Sentence_Generation(15)


