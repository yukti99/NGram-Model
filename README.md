# NGram-Model
Python Implementation of N-Gram Model in Natural Language Processing. A statistical language model is a probability distribution over sequences of words. Given
such a sequence, say of length m, it assigns a probability P(w1, w1, w2, w3, ......,wn ) to the whole sequence. 
This program assigns a probability to a sentence using n-gram modeling using a collection of documents.

  1. The value of N can be given by the user so that n-grams of any size could be used for probabilistic language modeling.  
  2. Code uses documents 1,2,3 to test the N-Gram Model
  3. LanguageModelling.py has the main model class that is used in model_driver.py
  4. It also implements additional functionalities like Add-1 smoothing to demonstrate language model’s capability to handle various issues like zero probability.
  5. This program also generates probable sentence given a particular text corpus and use it to calculte the likelihood of a test sentence.


To Execute:

	python3 model_driver.py
	<enter the value of 'N'>

Output:

	Will be displayed on the terminal 


