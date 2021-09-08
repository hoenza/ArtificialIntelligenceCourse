from sys import stdin
from math import sqrt
from copy import deepcopy as cpy
import heapq
from functools import total_ordering

tableD = []
for line in stdin:
    if line == "":
        break
    if line[-1] == '\n':
        tableD.append(list(line[:-1]))
    else:
        tableD.append(list(line))

@total_ordering
class State:

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return 0
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return 1
        return NotImplemented

    def __init__(self, table):
        self.table = table
        self.ambulance = [-1, -1]
        self.script = ""
        self.heur = 0
        for i in range(len(self.table)):
            for j in range(len(self.table[0])):
                if self.table[i][j] == 'A':
                    self.table[i][j] = ' '
                    self.ambulance = [i, j]
                    self.setString()
        self.father = -1
        self.cost = 0
    
    def getPatients(self):
        patients = []
        for i in range(len(self.table)):
            for j in range(len(self.table[i])):
                if self.table[i][j] == 'P':
                    patients.append([i, j])
        return patients
    
    def numPatients(self):
        ans = 0
        for i in range(len(self.table)):
            for j in range(len(self.table[i])):
                if self.table[i][j] == 'P':
                    ans += 1
        return ans
    
    def getHospitals(self):
        hospitals = []
        for i in range(len(self.table)):
            for j in range(len(self.table[i])):
                if ord('0') < ord(self.table[i][j]) < ord('4'):
                    hospitals.append([i, j])
        return hospitals
    
    def numHospitals(self):
        ans = 0
        for i in range(len(self.table)):
            for j in range(len(self.table[i])):
                if ord('0') < ord(self.table[i][j]) < ord('4'):
                    ans += ord(self.table[i][j]) - ord('0')
        return ans

    def isGoal(self):
        return self.numPatients() == 0 or self.numHospitals() == 0
    
    def equal(self, other):
        if self.table != other.table:
            return False
        if self.ambulance != other.ambulance:
            return False
        return True
    
    def checkHost(self, target):
        return ord('0') < ord(self.table[target[0]][target[1]]) < ord('4')

    def genChilds(self):
        childs = []
        newChild = self.genNeighbour('up')
        if newChild != -1:
            childs.append(newChild)
        newChild = self.genNeighbour('down')
        if newChild != -1:
            childs.append(newChild)
        newChild = self.genNeighbour('left')
        if newChild != -1:
            childs.append(newChild)
        newChild = self.genNeighbour('right')
        if newChild != -1:
            childs.append(newChild)
        return childs

    def setString(self):
        self.script = ""
        for line in self.table:
            self.script += "".join(line)
        self.script += "".join([str(i) for i in self.ambulance])

    def genNeighbour(self, direction):
        i, j = self.ambulance
        if direction == 'up':
            i -= 1
        if direction == 'down':
            i += 1
        if direction == 'left':
            j -= 1
        if direction == 'right':
            j += 1
        
        if self.table[i][j] == '#':
            return -1
        
        if self.table[i][j] == ' ' or self.table[i][j] == '0' or self.checkHost([i, j]):
            child = State(cpy(self.table))
            child.ambulance = [i, j]
            child.father = self
            child.cost = self.cost + 1
            child.setString()
            return child
        
        if self.table[i][j] == 'P':
            if direction == 'up' and (self.table[i-1][j] == ' ' or self.table[i-1][j] == '0'):
                newTable = cpy(self.table)
                newTable[i-1][j] = 'P'
                newTable[i][j] = ' '
                child = State(newTable)
                child.ambulance = [i, j]
                child.father = self
                child.cost = self.cost + 1
                child.setString()
                return child
            if direction == 'up' and self.checkHost([i-1, j]):
                newTable = cpy(self.table)
                newTable[i-1][j] = chr(ord(self.table[i-1][j])-1)
                if newTable[i-1][j] == '0':
                    newTable[i-1][j] = ' '
                newTable[i][j] = ' '
                child = State(newTable)
                child.ambulance = [i, j]
                child.father = self
                child.cost = self.cost + 1
                child.setString()
                return child
            if direction == 'down' and (self.table[i+1][j] == ' ' or self.table[i+1][j] == '0'):
                newTable = cpy(self.table)
                newTable[i+1][j] = 'P'
                newTable[i][j] = ' '
                child = State(newTable)
                child.ambulance = [i, j]
                child.father = self
                child.cost = self.cost + 1
                child.setString()
                return child
            if direction == 'down' and self.checkHost([i+1, j]):
                newTable = cpy(self.table)
                newTable[i+1][j] = chr(ord(self.table[i+1][j])-1)
                if newTable[i+1][j] == '0':
                    newTable[i+1][j] = ' '
                newTable[i][j] = ' '
                child = State(newTable)
                child.ambulance = [i, j]
                child.father = self
                child.cost = self.cost + 1
                child.setString()
                return child
            if direction == 'left' and (self.table[i][j-1] == ' ' or self.table[i][j-1] == '0'):
                newTable = cpy(self.table)
                newTable[i][j-1] = 'P'
                newTable[i][j] = ' '
                child = State(newTable)
                child.ambulance = [i, j]
                child.father = self
                child.cost = self.cost + 1
                child.setString()
                return child
            if direction == 'left' and self.checkHost([i, j-1]):
                newTable = cpy(self.table)
                newTable[i][j-1] = chr(ord(self.table[i][j-1])-1)
                if newTable[i][j-1] == '0':
                    newTable[i][j-1] = ' '
                newTable[i][j] = ' '
                child = State(newTable)
                child.ambulance = [i, j]
                child.father = self
                child.cost = self.cost + 1
                child.setString()
                return child
            if direction == 'right' and (self.table[i][j+1] == ' ' or self.table[i][j+1] == '0'):
                newTable = cpy(self.table)
                newTable[i][j+1] = 'P'
                newTable[i][j] = ' '
                child = State(newTable)
                child.ambulance = [i, j]
                child.father = self
                child.cost = self.cost + 1
                child.setString()
                return child
            if direction == 'right' and self.checkHost([i, j+1]):
                newTable = cpy(self.table)
                newTable[i][j+1] = chr(ord(self.table[i][j+1])-1)
                if newTable[i][j+1] == '0':
                    newTable[i][j+1] = ' '
                newTable[i][j] = ' '
                child = State(newTable)
                child.ambulance = [i, j]
                child.father = self
                child.cost = self.cost + 1
                child.setString()
                return child
        
        return -1
         
    def print(self):
        print()
        print('\   0    1    2    3    4    5    6    7    8    9    ')
        for num, line in enumerate(self.table):
            print(num, line)
        print(self.ambulance, '>>>>>>>>>>>>>>>>>>>>>>')
    
    def findInList(self, list):
        for i in range(len(list)):
            if self.equal(list[i]):
                return i
        return -1

    def setHeur1(self):
        self.heur = self.numPatients() + self.numHospitals()

    def findDistanceH(self, loc):
        pi, pj = loc
        minVal = 1000
        minH = -1
        for i in range(len(self.table)):
            for j in range(len(self.table[i])):
                if self.checkHost([i, j]) and sqrt((pi-i)**2 + (pj-j)**2) < minVal:
                    minVal = sqrt((pi-i)**2 + (pj-j)**2)
                    minH = [i, j]
        return minVal

    def setHeur2(self):
        self.heur = 0
        patients = self.getPatients()
        for patient in patients:
            self.heur += self.findDistanceH(patient)

    def f(self):
        return self.cost + self.heur   



def BFS(intialState):
    visited = set([])

    frontier = []
    frontier.append(initialState)

    frontierCoded = set([])
    frontierCoded.add(intialState.script)

    doubleStates = 0

    while len(frontier):
        cur = frontier.pop(0)
        frontierCoded.remove(cur.script)
        visited.add(cur.script)

        if cur.isGoal():
            return [cur.cost, len(visited), doubleStates]
        
        newChilds = cur.genChilds()
        for child in newChilds:
            if child.script in visited:
                doubleStates += 1
            elif child.script in frontierCoded:
                pass
            else:
                frontier.append(child)
                frontierCoded.add(child.script)
    
    return [-1, -1, -1]


def DFS(visited, doubleStates, state, depth):
    frontier = []
    frontierSet = set([])
    frontier.append(state)
    frontierSet.add(state.script)

    localV = {}
    while len(frontier):
        cur = frontier.pop()
        localV[state.script] = state.cost

        if cur.script in visited:
            doubleStates[0] += 1
        else:
            visited.add(cur.script)
            if cur.isGoal():
                return [cur.cost, len(visited), doubleStates[0]]

        if cur.cost >= depth:
            continue
    
        newChilds = cur.genChilds()
        for child in newChilds:
            if child.script not in localV or localV[child.script] > child.cost:
                localV[child.script] = child.cost
                frontier.append(child)
        
    return [-1, -1, -1]

def IDS(initialState):
    depth = 0
    visited = set([])
    doubleStates = [0]
    while True:
        result = DFS(visited, doubleStates, initialState, depth)
        if result[0] != -1:
            return result
        depth += 1
 


def Astr1(initialState):
    frontier = []
    visited = set([])
    initialState.setHeur1()
    heapq.heappush(frontier, (initialState.f(), initialState))
    frontierSet = {}
    frontierSet[initialState.script] = initialState
    doubleStates = 0
    while len(frontier):
        f , cur = heapq.heappop(frontier)
        if cur.father == -2:
            continue
        frontierSet.pop(cur.script)
        visited.add(cur.script)

        if cur.isGoal():
            return [cur.cost, len(visited), doubleStates]
        
        newChilds = cur.genChilds()
        for child in newChilds:
            child.setHeur1()
            if child.script in visited:
                doubleStates += 1
            elif child.script in frontierSet:
                if child.cost < frontierSet[child.script].cost:
                    preChild = frontierSet[child.script]
                    preChild.father = -2
                    heapq.heappush(frontier, (child.f(), child))
                    frontierSet[child.script] = child
            else:
                heapq.heappush(frontier, (child.f(), child))
                frontierSet[child.script] = child
    
    return [-1, -1, -1]

def Astr2(initialState):
    frontier = []
    visited = set([])
    initialState.setHeur2()
    heapq.heappush(frontier, (initialState.f(), initialState))
    frontierSet = {}
    frontierSet[initialState.script] = initialState
    doubleStates = 0
    while len(frontier):
        f , cur = heapq.heappop(frontier)
        if cur.father == -2:
            continue
        frontierSet.pop(cur.script)
        visited.add(cur.script)

        if cur.isGoal():
            return [cur.cost, len(visited), doubleStates]
        
        newChilds = cur.genChilds()
        for child in newChilds:
            child.setHeur2()
            if child.script in visited:
                doubleStates += 1
            elif child.script in frontierSet:
                if child.cost < frontierSet[child.script].cost:
                    preChild = frontierSet[child.script]
                    preChild.father = -2
                    heapq.heappush(frontier, (child.f(), child))
                    frontierSet[child.script] = child
            else:
                heapq.heappush(frontier, (child.f(), child))
                frontierSet[child.script] = child
    
    return [-1, -1, -1]

   
initialState = State(tableD)

ans = BFS(initialState)
print("BFS algorithm")
print("distance             |", ans[0])
print("num of states        |", ans[1] + ans[2])
print("num of unique states |", ans[1])
print()

ans = IDS(initialState)
print("IDS algorithm")
print("distance             |", ans[0])
print("num of states        |", ans[1] + ans[2])
print("num of unique states |", ans[1])
print()

ans = Astr1(initialState)
print("A* heuristic1")
print("distance             |", ans[0])
print("num of states        |", ans[1] + ans[2])
print("num of unique states |", ans[1])
print()

ans = Astr2(initialState)
print("A* heurisitc2")
print("distance             |", ans[0])
print("num of states        |", ans[1] + ans[2])
print("num of unique states |", ans[1])
print()