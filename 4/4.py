import nltk
cfg_rules = """
S -> NP-SBJ VP STOP 
NP -> CD JJ NN | CD JJ NNS
NP-SBJ -> DT NN NN
VP -> VBZ NP

DT -> 'the' | 'a'
NN -> 'purchase' | 'price' | 'guild' | 'strike'
VBZ -> 'includes' | 'began'
CD -> 'two'
JJ -> 'ancillary'
NNS -> 'companies'
STOP -> '.'
"""
grammar  = nltk.CFG.fromstring(cfg_rules)

cfg_rules = """
S -> NP-SBJ VP STOP | NP NP STOP
NP -> CD JJ NN | CD JJ NNS | NP PP-DIR | DT NN | DT NN CC NN NN | NNP CD | DT NN VBD
PP-DIR -> IN NP
NP-SBJ -> DT NN NN | DT NN
VP -> VBZ NP | VBD NP PP-TMP
PP-TMP -> IN NP


DT -> 'the' | 'a'
NNP -> 'March'
NN -> 'purchase' | 'price' | 'guild' | 'strike' | 'TV' | 'movie' | 'industry' | 'company'
VBZ -> 'includes'
VBD -> 'began' | 'bought'
CD -> 'two' | '1988' | 'one'
CC -> 'and'
JJ -> 'ancillary'
NNS -> 'companies'
IN -> 'in' | 'against'
STOP -> '.'
"""
grammar = nltk.CFG.fromstring(cfg_rules)

grammar.is_flexible_chomsky_normal_form()

sentences = [
    "the purchase price includes two ancillary companies .".split(),
    "the guild began a strike against the TV and movie industry in March 1988 .".split(),
]
for s in sentences:
    grammar.check_coverage(s)
    
cfg_rules = """
S -> S1 STOP
S1 -> NP-SBJ VP
NP -> NP1 NN | NP1 NNS | NNP CD | NP PP-DIR | NP2 NN
NP1 -> CD JJ
NP2 -> NP3 NN
NP3 -> NP4 CC
NP4 -> DT NN
PP-DIR -> IN NP
NP-SBJ -> NP-SBJ1 NN | DT NN
NP-SBJ1 -> DT NN
VP -> VBZ NP | VP1 PP-TMP
VP1 -> VBD NP
PP-TMP -> IN NP

DT -> 'the' | 'a'
NNP -> 'March'
NN -> 'purchase' | 'price' | 'guild' | 'strike' | 'TV' | 'movie' | 'industry' | 'company'
VBZ -> 'includes'
VBD -> 'began' | 'bought'
CD -> 'two' | '1988' | 'one'
CC -> 'and'
JJ -> 'ancillary'
NNS -> 'companies'
IN -> 'in' | 'against'
STOP -> '.'
"""

grammar = nltk.CFG.fromstring(cfg_rules)

sentences = [
    "the purchase price includes two ancillary companies .".split(),
    "the guild began a strike against the TV and movie industry in March 1988 .".split(),
    'the guild bought one ancillary company .'.split()
]
for s in sentences:
    grammar.check_coverage(s)

print(grammar.is_flexible_chomsky_normal_form())
print(grammar.is_chomsky_normal_form())

from nltk.parse.chart import BottomUpChartParser
parser = BottomUpChartParser(grammar)

sentences = ['the purchase price includes two ancillary companies .'.split(),
                'the guild began a strike against the TV and movie industry in March 1988 .'.split(),
                "the purchase price includes two ancillary companies .".split(),
                'the guild bought one ancillary company .'.split()]

for sent in sentences:
    for p in parser.parse(sent):
        p.draw()
        
from nltk.corpus import treebank
print(treebank.parsed_sents()[0])
print(treebank.parsed_sents()[1])

from nltk.grammar import CFG, Nonterminal

prods = list({production for sent in treebank.parsed_sents() for production in sent.productions()})
t_grammar = CFG(Nonterminal('S'), prods)

sents = [
    'Mr. Vinken is chairman .'.split(),
    'Stocks rose .'.split(),
    'Alan introduced a plan .'.split()
]

t_parser = BottomUpChartParser(t_grammar)

    
parses = 0
for s in sents[:1]:
    for p in t_parser.parse(s):
        if parses < 5:
            print(p)
        parses += 1
        
print(parses)

transitions = {}
total = 0

for sent in treebank.parsed_sents():
    for parse in sent:
        for prod in parse.productions():
            if prod.lhs() == Nonterminal('S'):
                total += 1
                if transitions.get(prod.rhs()):
                    transitions[prod.rhs()] += 1
                else:
                    transitions[prod.rhs()] = 1

filt_trans = {}
over_5 = 0
for k, v in transitions.items():
    if v >= 5:
        filt_trans[k] = (v, v / total)
    
filt_trans = {k: (v, v/ over_5) for k, v in filt_trans.items()}

filt_trans

from nltk import induce_pcfg
from nltk import InsideChartParser

prods = list({production for sent in treebank.parsed_sents() for production in sent.productions()})
g_pfcg = induce_pcfg(Nonterminal('S'), prods)

p_parser = InsideChartParser(g_pfcg, beam_size=400)

sents = [
    'Mr. Vinken is chairman .'.split(),
    'Stocks rose .'.split(),
    'Alan introduced a plan .'.split()
]

for sent in sents:
    print(sent)
    for p in p_parser.parse(sent):
        print(p)
list(parse)
list(p_parser.parse(['you', 'are', 'sleeping']))