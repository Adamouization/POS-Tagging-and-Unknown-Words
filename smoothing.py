from nltk import FreqDist, WittenBellProbDist

emissions = [('N', 'apple'), ('N', 'apple'), ('N', 'banana'), ('Adj', 'apple'), ('V', 'sing')]
smoothed = {}
tags = set([t for (t,_) in emissions])
for tag in tags:
    words = [w for (t,w) in emissions if t == tag]
    smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)
print('probability of N -> apple is', smoothed['N'].prob('apple'))
print('probability of N -> banana is', smoothed['N'].prob('banana'))
print('probability of N -> peach is', smoothed['N'].prob('peach'))
print('probability of V -> sing is', smoothed['V'].prob('sing'))
print('probability of V -> walk is', smoothed['V'].prob('walk'))
