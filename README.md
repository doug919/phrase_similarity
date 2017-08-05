This tool measures similarity scores between phrases. Two pre-trained embedding models are included: *CNNPhrase* and *WAVG*.
The *CNNPhrase* [1] is a phrase similarity model that is trained on the Paraphrase Database [4] with a Convolutional Neural Network (CNN) architecture to model sentence structure.
And the *WAVG* simply averages the paraphrase word embeddings [2]. 
The models can be appied in diverse phrase-similarity-related tasks, e.g., [3] uses our models to identify similar phrases on a Twitter dataset.


# 1. Install dependencies:
    
## 1.1 install virtualenv   
Download the virtualenv tool: 

    $ git clone https://github.com/pypa/virtualenv

go to the downloaded folder and create a virtualenv. You can specifity the environment name by replacing my_env_name in the following commands:
    
    $ python virtualenv.py my_env_name
        
Set the virtualenv up

    $ source my_env_name/bin/activate
    
You should see that (my_env_name) appeared in your command line.
    
    
## 1.2 install theano
since the default theano version is deprecated, you need to install a newer version. Fortunately, the one from pip is workable. So just run the following command to install the theano.
    
    (my_env_name)$ pip install theano
    
# 2. Set this repository up
Download this repository to the folder you prefer:

    (my_env_name)$ git clone https://lee2226@gitlab.com/lee2226/phrase_similarity.git

Make sure you replace lee2226 with your account. Go into the directory:

    (my_env_name)$ cd phrase_similarity

Download the model file from here: https://drive.google.com/open?id=0B9nkB1SpB_xLX0hoWVNDQnJ0LVU 
and then upload it to the repo folder.

Decompress the model file:
    
      (Upload the model file to this folder)
    (my_env_name)$ tar xzvf phrase_models.tar.gz
    
# 3. Run sim.py
Now you should be able to the program with the following command:

    (my_env_name)$ python sim.py example_input.txt CNN

or
    
    (my_env_name)$ python sim.py example_input.txt WAVG
    
,where the example_input.txt is the input file that contains alll phrase pairs; the CNN and WAVG is for specifying the model type that is used for measuring the similarity. The CNN is a pre-trained convolutional neural model, and the WAVG simply averages the paraphrase word embeddings.

It might takes few minutes to run, even if you only put few examples in the input file, since it needs to load the large model files.

The output is shown on the stdout. You can pipe them into a file, like this:

    (my_env_name)$ python sim.py example_input.txt CNN > output_file


# 4. Others
If you want to check the detailed command descriptions, you can type:
    
    (my_env_name)$ python sim.py -h
    usage: sim.py [-h] [-v] [-d] INPUT_FILE MODEL_TYPE

    Measure similarity scores between phrases

    positional arguments:
        INPUT_FILE     input phrases.
        MODEL_TYPE     CNN or WAVG (word average)

    optional arguments:
        -h, --help     show this help message and exit
        -v, --verbose  show info messages
        -d, --debug    show debug messages

# References

[1] I-Ta Lee, Mahak Goindani, Chang Li, Di Jin, Kristen Marie Johnson, Xiao Zhang, Maria Leonor Pacheco, and Dan Goldwasser, PurdueNLP at SemEval-2017 Task 1: Predicting Semantic Textual Similarity with Paraphrase and Event Embeddings, SemEval 2017

[2] John Wieting, Mohit Bansal, Kevin Gimpel, and Karen Livescu, Towards Universal Paraphrastic Sentence Embeddings, ICLR 2016

[3] Kristen Marie Johnson, I-Ta Lee, and Dan Goldwasser, Ideological Phrase Indicators for Classification of Political Discourse
Framing on Twitter, NLP+CSS 2017

[4] Juri Ganitkevitch, Benjamin Van Durme, and Chris Callison-Burch, PPDB: The Paraphrase Database, HLT-NAACL 2013
