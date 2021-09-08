import csv
import nltk
nltk.download('wordnet')
from math import log
import random
from nltk.stem import PorterStemmer, SnowballStemmer


def read_csv(file_name):
    fields = []
    rows = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        fields = next(csv_reader)
        for row in csv_reader:
            rows.append(row)
        for row in rows:
            row[4] = clean_text(row[4])
            row[6] = clean_text(row[6])
    return fields, rows

def read_csv_test(file_name):
    fields = []
    rows = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        fields = next(csv_reader)
        for row in csv_reader:
            rows.append(row)
        for row in rows:
            row[1] = clean_text(row[1])
            row[4] = clean_text(row[4])
    return fields, rows

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if not word in nltk.corpus.stopwords.words("english")]
    lemma = nltk.WordNetLemmatizer()
    tokens = [lemma.lemmatize(word, pos="v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos="n") for word in tokens]
    # porter = PorterStemmer()
    # for i in range(len(tokens)):
    #     tokens[i] = porter.stem(tokens[i])
    return tokens

def seperate_data(data):
    train_data = []
    valid_data = []
    dictNum = {}
    for row in data:
        if not row[2] in dictNum:
            dictNum[row[2]] = [1]
        else:
            dictNum[row[2]][0] += 1
    for row in data:
        if len(dictNum[row[2]]) < 2:
            dictNum[row[2]].append(1)
            train_data.append(row)
        elif dictNum[row[2]][1] < dictNum[row[2]][0] *(8/10):
            dictNum[row[2]][1] += 1
            train_data.append(row)
        else:
            valid_data.append(row)
    return train_data, valid_data

def over_sample(data):
    numClasses = [0, 0, 0]
    for row in data:
        if row[2] == 'BUSINESS':
            numClasses[0] += 1
        elif row[2] == 'TRAVEL':
            numClasses[1] += 1
        else:
            numClasses[2] += 1
    
    maxClass = max(numClasses)
    diff = maxClass - numClasses[0]
    newData = []
    while len(newData) < diff:
        s = random.choice(data)
        if s[2] == 'BUSINESS':
            newData.append(s)
    for row in newData:
        data.append(row)
    
    diff = maxClass - numClasses[1]
    newData = []
    while len(newData) < diff:
        s = random.choice(data)
        if s[2] == 'TRAVEL':
            newData.append(s)
    for row in newData:
        data.append(row)
    
    diff = maxClass - numClasses[2]
    newData = []
    while len(newData) < diff:
        s = random.choice(data)
        if s[2] == 'STYLE & BEAUTY':
            newData.append(s)
    for row in newData:
        data.append(row)
    random.shuffle(data)
    return data

def confusion_matrix(truth_labels):
    ans = {}
    for row in truth_labels:
        if not row[0] in ans:
            ans[row[0]] = {}
        else:
            if not row[1] in ans[row[0]]:
                ans[row[0]][row[1]] = 1
            else:
                ans[row[0]][row[1]] += 1
    return ans

class Model:
    def __init__(self, file_name):
        self.data_fields, data = read_csv(file_name)
        data = over_sample(data)
        self.train_data, self.valid_data = seperate_data(data)
        self.test_data = None
    
    def train0(self):
        self.dictionary = {}
        self.numClasses = [0, 0, 0]
        self.numWords = [0, 0, 0]
        for row in self.train_data:
            if row[2] == 'BUSINESS':
                self.numClasses[0] += 1
            elif row[2] == 'TRAVEL':
                self.numClasses[1] += 1
            else:
                continue
            
            for word in row[6]:
                if not word in self.dictionary:
                    self.dictionary[word] = [0, 0, 0]
                if row[2] == 'BUSINESS':
                    self.dictionary[word][0] += 1
                    self.numWords[0] += 1
                elif row[2] == 'TRAVEL':
                    self.dictionary[word][1] += 1
                    self.numWords[1] += 1

    def train(self):
        self.dictionary = {}
        self.numClasses = [0, 0, 0]
        self.numWords = [0, 0, 0]
        for row in self.train_data:
            if row[2] == 'BUSINESS':
                self.numClasses[0] += 1
            elif row[2] == 'TRAVEL':
                self.numClasses[1] += 1
            elif row[2] == 'STYLE & BEAUTY':
                self.numClasses[2] += 1
            
            for word in row[6]:
                if not word in self.dictionary:
                    self.dictionary[word] = [0, 0, 0]
                if row[2] == 'BUSINESS':
                    self.dictionary[word][0] += 1
                    self.numWords[0] += 1
                elif row[2] == 'TRAVEL':
                    self.dictionary[word][1] += 1
                    self.numWords[1] += 1
                elif row[2] == 'STYLE & BEAUTY':
                    self.dictionary[word][2] += 1
                    self.numWords[2] += 1

    def label(self, datan, phase):
        if datan == "train":
            data = self.train_data
        elif datan == "valid":
            data = self.valid_data
        else:
            data = self.test_data
        
        labels = []
        for row in data:
            score1 = log(1)
            score2 = log(1)
            score3 = log(1)
            nameClasses = ['BUSINESS', 'TRAVEL', 'STYLE & BEAUTY']

            if datan == "test":
                sentence = row[4]
            else:
                sentence = row[6]

            for word in sentence:
                if not word in self.dictionary:
                    continue
                if self.dictionary[word][0] == 0:
                    score1 -= 10
                else:
                    score1 += log((self.dictionary[word][0])/self.numWords[0])
                if self.dictionary[word][1] == 0:
                    score2 -= 10
                else:
                    score2 += log((self.dictionary[word][1])/self.numWords[1])
                if phase == 1:
                    if self.dictionary[word][2] == 0:
                        score3 -= 10
                    else:
                        score3 += log((self.dictionary[word][2])/self.numWords[2])
            
            score1 += log((self.numClasses[0])/sum(self.numClasses))
            score2 += log((self.numClasses[1])/sum(self.numClasses))
            if phase == 1:
                score3 += log((self.numClasses[2])/sum(self.numClasses))

            if phase == 0:
                tmp = [score1, score2]
                labels.append(nameClasses[tmp.index(max(tmp))])
            else:
                tmp = [score1, score2, score3]
                labels.append(nameClasses[tmp.index(max(tmp))])
        return labels
            
    def scores(self, data, phase):
        def count_correct_labels(truth_labels, label):
            count = 0
            for row in truth_labels:
                if row[0] == row[1] and row[0] == label:
                    count +=1
            return count
        
        def count_detected_labels(truth_labels, label, phase):
            count = 0
            for row in truth_labels:
                if phase==0 and row[0] == 'STYLE & BEAUTY':
                    continue
                if row[1] == label:
                    count += 1
            return count   
        
        def count_labels(truth_labels, label):
            count = 0
            for row in truth_labels:
                if row[0] == label:
                    count += 1
            return count
        
        def count_trues(truth_labels, phase):
            count = 0
            all = 0
            for row in truth_labels:
                if phase == 0 and row[0] == 'STYLE & BEAUTY':
                    continue
                all += 1
                if row[0] == row[1]:
                    count += 1
            return count, all
        
        labels = self.label(data, phase)

        if data == "train":
            data = self.train_data
        elif data == "valid":
            data = self.valid_data
        else:
            data = self.test_data
        

        recalls = {'BUSINESS':0, 'TRAVEL':0, 'STYLE & BEAUTY':0}
        precisions = {'BUSINESS':0, 'TRAVEL':0, 'STYLE & BEAUTY':0}
        accuracys = 0

        truth_labels = []
        for i in range(len(data)):
            tmp = []
            tmp.append(data[i][2])
            tmp.append(labels[i])
            truth_labels.append(tmp)
        confusion = confusion_matrix(truth_labels)
        print("confusion matrix ")
        print(confusion)
        recalls['BUSINESS'] = count_correct_labels(truth_labels, 'BUSINESS')/count_labels(truth_labels, 'BUSINESS')
        precisions['BUSINESS'] = count_correct_labels(truth_labels, 'BUSINESS')/count_detected_labels(truth_labels, 'BUSINESS', phase)
        recalls['TRAVEL'] = count_correct_labels(truth_labels, 'TRAVEL')/count_labels(truth_labels, 'TRAVEL')
        precisions['TRAVEL'] = count_correct_labels(truth_labels, 'TRAVEL')/count_detected_labels(truth_labels, 'TRAVEL', phase)
        
        if phase==1:
            recalls['STYLE & BEAUTY'] = count_correct_labels(truth_labels, 'STYLE & BEAUTY')/count_labels(truth_labels, 'STYLE & BEAUTY')
            precisions['STYLE & BEAUTY'] = count_correct_labels(truth_labels, 'STYLE & BEAUTY')/count_detected_labels(truth_labels, 'STYLE & BEAUTY', phase)

        trues, alll = count_trues(truth_labels, phase)
        accuracys = trues/alll
        return recalls, precisions, accuracys

    def test(self, file_name):
        data_fields, self.test_data = read_csv_test(file_name)
        labels = self.label("test", 1)
        with open('output.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['index', 'category'])
            for i,row in enumerate(labels):
                writer.writerow([i, row])

        

               

model = Model("data.csv")
# model.train0()
# a, b, c = model.scores("valid", 0)
# print("Recall", a)
# print("Precision", b)
# print("Accuracy", c)

model.train()
a, b, c = model.scores("valid", 1)
print("Recall", a)
print("Precision", b)
print("Accuracy", c)
model.test("test.csv")

