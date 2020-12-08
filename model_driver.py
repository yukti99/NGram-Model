# -*- coding: utf-8 -*-
"""
@author: yukti
"""


from LanguageModelling import NGramModel



# Getting Data/Corpus to use for N-Gram model
N = 3
N = int(input("Please enter the value of N for the N-Gram Model = "))
doc = open("doc1.txt","r")

data = "Ronald read her book. I read a different book. Jake read a book by Twain! "
test_sentence = "Ronald read a book"

data2 = doc.read()
test_sentence2 = "compare this increased efficiency of hours"


model_obj = NGramModel(N,data2)
model_obj.Language_Modelling(test_sentence2)
model_obj.Sentence_Generation(15)


