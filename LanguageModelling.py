# -*- coding: utf-8 -*-
"""
@author: yukti

"""

# importing python libraries
import re 
import numpy as np
import string 
import nltk
from textblob import TextBlob
import math 



class NGramModel:
    Vocab_size = 0
    ngram_probs = []
    def __init__(self, N, data):
        self.N = N
        self.data = data
        
   
    """ PREPROCESSING / CLEANING RAW DATA """
    def Preprocess_Text(self,text):
        text = text.lower()
        text = "".join(ch for ch in text if not ch.isdigit())
        punct = """!\"#$%&'()*+-/:;<=>?@[\\]^_`{|}~"""
        text = "".join([ch for ch in text if ch not in punct])
        text = re.sub(' +|\\n+', ' ', text)
        text = re.sub('<[^<]+?>', '', text)
        text = re.sub("\s\s+" , " ", text)
        tokenized_text = nltk.word_tokenize(text)
        for i in range(len(tokenized_text)):
            sp = TextBlob(tokenized_text[i])
            tokenized_text[i] = str(sp.correct())
        text = " ".join(tokenized_text)            
        return text
    
    """ FINDING NUMBER OF UNIQUE WORDS IN THE TEXT """
    def get_vocab_size(self,text):    
        text = nltk.word_tokenize(text)
        v = []
        for i in text:
            if i not in v:
                v.append(i)
        return len(v)
    
    """ PREPARING DATA FOR USING NGRAM MODEL """
    def prepare_data(self,data):
        
        # PREPROCESSING THE TEXT / TEXT CLEANING
        text = self.Preprocess_Text(data)
        NGramModel.Vocab_size = self.get_vocab_size(text)
        print("Vocabulary Size = ",NGramModel.Vocab_size)
        
        # CONVERTING THE TEXT INTO A LIST OF SENTENCES
        text = text.split(".")
        for i in range(0,len(text)):
            if (text[i]==''):
                text.pop(i)
                  
        # ADDING <S> AT START AND </S> AT END OF EACH SENTENCE OF TEXT
        cleaned_text = []    
        for sentence in text:
            s = ["<s>"]+sentence.strip().split()+["</s>"]
            cleaned_text.append(s)
        
        return cleaned_text 
    
    """ STORING COUNTS AND PROBABILITIES OF NGRAMS IN THE TRAINING TEXT """
    def evaluate_Ngrams(self,text):
        ngrams = []
        ngramCounts = []
        for n in range(self.N):
            # dictionary for storing count of ngrams
            currGramCounts = {}
            totalCount = 0
            for line in text:
                for i in range(len(line) - (n)):
                    # grouping of words according the 'n' value for ngram model
                    ngram = line[i:i+n+1]
                    ngram = " ".join(ngram)
                    if not ngram in currGramCounts:
                        currGramCounts[ngram] = 0
                    currGramCounts[ngram] += 1
                    # total count of ngrams
                    totalCount += 1
            ngramCounts.append(currGramCounts)
            
            # dictionaries to store the current and overall Ngram probabilities 
            currProb = {}
            for ngram in currGramCounts.keys():
                ngram_prefix = " ".join(ngram.split()[:-1])
                if n != 0:
                    # this is C(wi-1), count of prefix sentence of the given word
                    countPrefix = ngramCounts[n - 1][ngram_prefix]
                else: 
                    # this is unigram model as n=0
                    countPrefix = totalCount 
                currProb[ngram] = ngramCounts[n][ngram] / (countPrefix*1.0)

            NGramModel.ngram_probs.append(currProb)        
               
        return ngramCounts
   
        
        

    """ EVALUATING PROBABILITY OF EACH WORD TO GET TEST SENTENCE PROBABILITY """
    def find_probability(self,ngrams,test_sentence):
        print("\nProbability of NGrams: \n")
        sentence_prob = 1.0
        sentence = ["<s>"] +test_sentence.split() + ["</s>"]
        for i in range(len(sentence) - (self.N - 1)):
            # Constructing ngrams from the text sentence to find the probability of given test sentence 
            # according the calculated ngram probabilities as given above
            ngram = sentence[i:i+self.N]
            ngram = " ".join(ngram)
            ngram_prefix = " ".join(ngram.split()[:-1])
            try:
                prefix_count = ngrams[self.N-2][ngram_prefix]
            except:
                prefix_count = 0
                
            # Applying Laplace smoothing (Add-one) to resolve if an Ngram doesn't appear in the training corpus
            try:
                ngram_prob = ngrams[self.N-1][ngram]
            except:
                ngram_prob = 0
                
            # calculating the probability of current Ngram 
            curr_ngram_prob = (ngram_prob + 1)*1.0 / (prefix_count+NGramModel.Vocab_size)                  
                
            # printing the probability of each ngram according to N (unigram, bigram, trigram, etc)        
            print("Probability of n-gram \"" + ngram + "\" = " + str(curr_ngram_prob))
            
            # multiplying the probabilities of each ngram to get overall sentence probability    
            sentence_prob *= curr_ngram_prob                   
        
        return sentence_prob
 
    """ Calculating Model Perplexity """
    def model_perplexity(self,test_sentence,sentence_prob):
        #if(sentence_prob!=0):
        N = len(test_sentence.split(" "))
        power = -1.0/N
        try:
            Perplexity = math.pow(sentence_prob,power)
            return Perplexity
        except:
            return -1
            
    """ ARRANGING NGRAM PROBABILITIES BY PREFIX FOR EASY ACCESS """
    def Ngrams_by_prefix(self,ngrams):
        ngramsByPrefix = {}
        for N in range(1,self.N+1):
            for ngram in ngrams[N - 1].keys():
                # get the first part of the ngram 
                prefix = " ".join(ngram.split()[:-1])
                if prefix not in ngramsByPrefix:
                    ngramsByPrefix[prefix] = {}
                # creating a dictionary of 'ngram:probability' pairs for given prefix
                ngramsByPrefix[prefix][ngram] = ngrams[N - 1][ngram] 
        return ngramsByPrefix

    " TO GENERATE SENTENCES USING THE GIVEN MODEL"
    def Generate_sentences(self,num_generate):
        generated_sentences = []
        start = "<s>"
        # getting ngrams by prefix for easy generation of probable sentences from this model
        ngramsByPrefix = self.Ngrams_by_prefix(NGramModel.ngram_probs)
        x = num_generate
        while(x>0):
            sentence = start        
            # repeat till we encounter end of sentence symbol i.e. '</s>'
            while(sentence.split()[-1] != "</s>"):               
                prefix = " ".join(sentence.split()[-self.N+1:])
                # this is a dictionary of all probable next ngrams starting with the given prefix
                try:
                    next_ngrams = ngramsByPrefix[prefix]
                except:
                    break
                # getting only the keys of the dictionary as they contain probable ngrams 
                next_ngrams_list = list(next_ngrams.keys())
                next_probs = np.array(list(next_ngrams.values())) 
                next_probs = next_probs/np.sum(next_probs)
                # using numpy.random.choice(a, size=None, replace=True, p=None) to choose an ngram acc to probabilities of sample
                ngram_i = np.random.choice(next_probs.shape[0],1,p = next_probs)[0]   
                # now we have the index of next probable ngram, we can concatenate that to generate the complete sentence 
                suffix = " ".join(next_ngrams_list[ngram_i].split()[-1:])
                sentence += " "+suffix 
            
            # removing the <s> at beginning and </s> at the end
            sentence = " ".join(sentence.split()[1:-1])
            if(sentence!=""):
                x-=1
                generated_sentences.append(sentence)
        
        if (generated_sentences!=[]):
            return generated_sentences
        else:
            return -1
    
        
    """ TO FIND HOW LIKELY IS THE TEST SENTENCE TO BE GENERATED USING THE GIVEN MODEL """
    def get_likelihood_sentence(self,test_sentence):
        s_count = 0
        generated_sentences = self.Generate_sentences(1000)
        for sentence in generated_sentences:
             # to check the likelihood of the test sentence to be generated by this method
            if sentence == test_sentence:
                s_count += 1
            
        likelihood = s_count*1.0 / 1000
        return likelihood
    
    def Language_Modelling(self,test_sentence):
        text = self.prepare_data(self.data)
        ngram_counts = self.evaluate_Ngrams(text)
        sentence_prob = self.find_probability(ngram_counts,test_sentence)
        print("\nProbability of the test sentence \"" +  test_sentence + "\" = " + str(sentence_prob)+"\n")        
        Perplexity = self.model_perplexity(test_sentence,sentence_prob)
        print("Perplexity of this NGram Model with N =",self.N," is ",Perplexity)    
        likelihood = self.get_likelihood_sentence(test_sentence)
        print("\nLikelihood of the test sentence according to this model: " + str(likelihood))
        
    def Sentence_Generation(self,num_generate):
        generated_sentences = self.Generate_sentences(num_generate)
        if (generated_sentences==-1):
            print("Sorry Sentence generation not possible...")
        else:
            print("\nThe following sentences can be generated using given model: ")
            no=1
            for sentence in generated_sentences:
                print(no,") ",sentence)
                no+=1      
        print("\n\n\n")
    









