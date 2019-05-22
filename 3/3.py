import nltk
from nltk.corpus import masc_tagged, treebank
from nltk.tag import hmm
from ass3utils import train_unsupervised 
import random

def insert_word(words_by_tag, word):
    if words_by_tag.get(word, False):
        words_by_tag[word] += 1
    else:
        words_by_tag[word] = 1

def get_counts():
    words = {}
    vb_transitions = {}
    
    for sent in masc_tagged.tagged_sents():
        for idx, (word, tag) in enumerate(sent):
            if words.get(tag):
                insert_word(words[tag], word)
            else:
                words[tag] = {}
                insert_word(words[tag], word)
                
            if tag == 'VB' and idx < (len(sent) - 1):
                next_tag = sent[idx + 1][1]
                if vb_transitions.get(next_tag):
                    vb_transitions[next_tag] += 1
                else:
                    vb_transitions[next_tag] = 1
                
    return words, vb_transitions

def get_probabilities(words, transitions, target, word, word_tag):
    total_trans = sum(transitions.values())
    trans_prob = transitions[target] / total_trans
    
    word_prob = words[word_tag][word] / sum(words[word_tag].values())
    
    print(f'Probability of VB being followed by {target} is {trans_prob * 100} %')
    print(f'Probability of {word} within the tag {word_tag} is {word_prob * 100} %')
    
def tag_sents(model):
    sents = ['Once we have finished , we will go out .',
         'There is always room for more understanding between warring peoples .',
         'Evidently , this was one of Jud \'s choicest tapestries , for the noble emitted a howl of grief and rage and leaped from his divan .']

    sents2 = [
        'Misjoggle in a gripty hifnipork .',
        'One fretigy kriptog is always better than several intersplicks .',
        'Hello my friend can you tag some words ineptly'
    ]
    
    sents3 = [
        'Yesterday these fiends operated upon Doggo .',
        'For a time, his own soul and this brain - maggot struggled for supremacy .'
    ]
    
    sents4 = [
        'System that prevents problems with little nephews',
        'Trump, Angered by Investigations, Blows Up Meeting With Democrats',
        'She Had Stage 4 Lung Cancer, and a Mountain to Climb',
        'Business partnership agreements are written agreements which states the rights, responsibility, and accountability of the parties involved in the agreement',
        ]
    
    for sent in sents:
        print(model.tag(sent.split()), '\n')
    print('-' * 100)
    for sent in sents2:
        print(model.tag(sent.split()), '\n')
    print('-' * 100)
    for sent in sents3:
        print(model.tag(sent.split()), '\n')
    print('-' * 100)
    for sent in sents4:
        print(model.tag(sent.split()), '\n')

def log_prob(model):
    test_sents = [
        list(zip('Hi I am dog'.split(), [None] * 4)),
        list(zip('Try using your models as LMs'.split(), [None] * 6)),
        list(zip('Submit your answers'.split(), [None] * 3)),
        list(zip('Is you are we you they us them porridge'.split(), [None] * 9)),
        list(zip('Live computer eat slightly manic bag'.split(), [None] * 6)),
        list(zip('I am outputting a rather probable sentence but this one is still quite long one'.split(), [None] * 15)),
        list(zip('The the the the'.split(), [None] * 4)),
    ]
    
    for sent in test_sents:
        print(sent, 'Probability: ', model.log_probability(sent))
        
def sample_model(model):
    print(model.random_sample(random, 15))

train = hmm.HiddenMarkovModelTagger

model = train.train(masc_tagged.tagged_sents())

with open('radio_planet_tokens.txt') as radio:
    lines = radio.readlines()
    lines = list(map(lambda x: x.rstrip('\n').split(), lines))
    u_model = train_unsupervised(masc_tagged.tagged_sents(), lines, 10)

models = [model, u_model]

for m in models:
    tag_sents(m)
    log_prob(m)
    sample_model(m)
    print('W-W' * 100)