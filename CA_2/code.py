import re
import random
import string
import numpy
from numpy.random import choice
import heapq

library = open('global_text.txt', 'r').read()
data = open('encoded_text.txt', 'r').read()



class Decoder:
    def __init__(self, encoded_text):
        library = open('global_text.txt', 'r').read()
        self.dictionary = self.preprocess_text(library)
        self.encoded_text = encoded_text
        self.encoded_words = self.preprocess_text(encoded_text)
        self.generation = self.first_generation(100)
        self.scores = []
        self.weights = []
        self.goal = 0
        self.set_goal()
        self.set_weights()
        self.pc = 0.8
        self.pm = 0.4
        self.nm = 2
        return

    def preprocess_text(self, text):
        text = text.lower()
        new_text = ""
        for i in text:
            if ord('a') <= ord(i) <= ord('z'):
                new_text += i
            else:
                new_text += ' '
        words = set(new_text.split())
        return words

    def first_generation(self, num):
        ans = []
        for i in range(num):
            ans.append(''.join(random.sample(string.ascii_lowercase, 26)))
        return ans

    def score(self, gene):
        score = 0
        for word in self.encoded_words:
            dec_word = ""
            for char in word:
                dec_word += gene[ord(char) - ord('a')]
            
            if dec_word in self.dictionary:
                score += len(dec_word)
        return score

    def set_weights(self):
        scores = []
        for gene in self.generation:
            scores.append(self.score(gene))
        
        self.scores = scores
        sorted_scores = sorted(scores)
        weights = []
        sums = 0
        for score in scores:
            weights.append(sorted_scores.index(score)+1)
            sums += weights[-1]
        for i in range(len(weights)):
            weights[i] /= sums
        self.weights = weights
        return
    
    def cross(self):
        gene1 = choice(list(self.generation), p=self.weights)
        gene2 = choice(list(self.generation), p=self.weights)

        if choice([0, 1], p=[1-self.pc, self.pc]) == 0:
            return gene1, gene2

        indices = sorted(random.sample(range(26), 2))
        new_gene1 = list(26*' ')
        new_gene2 = list(26*' ')

        for i in range(indices[0], indices[1]+1):
            new_gene1[i] = gene2[i]
            new_gene2[i] = gene1[i]
        for i in range(26):
            if gene1[i] not in new_gene1:
                new_gene1[new_gene1.index(' ')] = gene1[i]
            if gene2[i] not in new_gene2:
                new_gene2[new_gene2.index(' ')] = gene2[i]
        return "".join(new_gene1), "".join(new_gene2)

    def mute(self, gene):
        if choice([0, 1], p=[1-self.pm, self.pm]) == 0:
            return gene
        gene = list(gene)
        for i in range(self.nm):
            ind0, ind1 = random.sample(range(26), 2)
            tmp = gene[ind0]
            gene[ind0] = gene[ind1]
            gene[ind1] = tmp
        return "".join(gene)

    def get_bests(self, percent):
        num = int(len(self.generation)*percent)
        h = []
        for i in self.generation:
            heapq.heappush(h, (-self.score(i), i))
        bests = []
        for i in range(num):
            sc, gen = heapq.heappop(h)
            bests.append(gen)
        return bests

    def update_generation(self):
        self.set_weights()
        new_gen = set([])
        bests = self.get_bests(0.3)
        for i in bests:
            new_gen.add(i)
        
        while len(new_gen) < len(self.generation):
            new_gene1, new_gene2 = self.cross()
            new_gene1 = self.mute(new_gene1)
            new_gene2 = self.mute(new_gene2)

            new_gen.add(new_gene1)
            new_gen.add(new_gene2)
        self.generation = new_gen
        self.set_weights()
        return
    
    def set_goal(self):
        goal = 0
        for word in self.encoded_words:
            goal += len(word)
        self.goal = goal
        return

    def best_gene(self):
        h = []
        for i in self.generation:
            heapq.heappush(h, (-self.score(i), i))
        sc, best = heapq.heappop(h)
        return best

    def run(self):
        while True:
            maxV = max(self.scores)
            if maxV == self.goal:
                best = self.best_gene()
                return best
            self.update_generation()

    def decode(self):
        best_gene = self.run()
        out = ""
        for chr in self.encoded_text:
            if ord('a') <= ord(chr) <= ord('z'):
                out += best_gene[ord(chr) - ord('a')]
            elif ord('A') <= ord(chr) <= ord('Z'):
                out += best_gene[ord(chr) - ord('A')].upper()
            else:
                out += chr
        return out
