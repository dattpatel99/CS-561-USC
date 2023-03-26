import random
import math
POPSIZE = 65
ITERATIONS = 2900
mutation_prob = 0.3
cross_prob = 1.0
alpha_mut = 0.01/10
alpha_cross = 0.05/50

class City:
    def __init__(self, coords):
        self.coords = coords

class Path:
    def __init__(self, path=[]):
        self.path = path

    def findDistance(self, cities):
        distance = 0
        for i in range(1, len(self.path)):
            distance += math.sqrt(pow((cities[self.path[i]].coords[0] - cities[self.path[i-1]].coords[0]), 2) + pow((cities[self.path[i]].coords[1] - cities[self.path[i-1]].coords[1]), 2) + pow((cities[self.path[i]].coords[2] - cities[self.path[i-1]].coords[2]), 2))
        distance += math.sqrt(pow((cities[self.path[0]].coords[0] - cities[self.path[-1]].coords[0]), 2) + pow((cities[self.path[0]].coords[1] - cities[self.path[-1]].coords[1]), 2) + pow((cities[self.path[0]].coords[2] - cities[self.path[-1]].coords[2]), 2))
        return distance

class Run:
    def __init__(self, population=[], cities=[], index=[], distances = [], fitness= [], mating_pairs=[], offsprings=[]):
        self.population = population
        self.cities = cities
        self.index = index
        self.distances = distances
        self.fitness = fitness
        self.maxDistance = -1
        self.best_path = -1
        self.total_fitness = -1
        self.mating_pairs = mating_pairs
        self.offsprings = offsprings
    
    def calculateDistance(self):
        self.distances = []
        for each in self.population:
            self.distances.append(round(each.findDistance(self.cities),4))
        self.maxDistance = max(self.distances)
        return min(self.distances)

    def calculateFitness(self):
        self.fitness = []
        tot_fitness= 0
        best_dis = 0
        for i in range(len(self.distances)):
            self.fitness.append(self.maxDistance - self.distances[i])
            tot_fitness += self.fitness[-1]
            if(best_dis < self.fitness[-1]):
                best_dis = self.fitness[-1]
                self.best_path = i
        self.total_fitness = tot_fitness
        return best_dis

    def selectParents(self):
        self.mating_pairs = []
        for i in range(POPSIZE//2):
            temp = []
            for i in range(2):
                stop = random.uniform(0, self.total_fitness)
                i = 0
                r = 0
                while r < stop:
                    r += self.fitness[i]
                    i += 1
                i -= 1
                temp.append(self.population[i])
            self.mating_pairs.append(temp)

    def crossOver(self):
        self.offsprings = []
        for i in range(len(self.mating_pairs)):
            p1 = self.mating_pairs[i][0]
            p2 = self.mating_pairs[i][1]
            if (random.random() < cross_prob):
                s = random.randrange(len(p1.path)//2)
                e = random.randint(s, len(p1.path)-1)
                c1 = Path(p1.path[:s] + p2.path[s:e] + p1.path[e:])
                c2 = Path(p2.path[:s] + p1.path[s:e] + p2.path[e:])
                self.offsprings.append(c1)
                self.offsprings.append(c2)
            else:
                self.offsprings.append(p1)
                self.offsprings.append(p2)

    def repairChildren(self):
        for i in range(len(self.offsprings)):
            leftout = [1 for k in range(len(self.cities))]
            repeat = []
            a_path = self.offsprings[i].path
            for j in range(len(a_path)):
                if(leftout[a_path[j]] == 1):
                    leftout[a_path[j]] = 0
                elif(leftout[a_path[j]] == 0):
                    repeat.append(j)
            
            for m in range(len(leftout)):
                if leftout[m] == 1:
                    a_path[repeat.pop()] = m

    def mutate(self):
        for i in range(len(self.offsprings)):
            if (random.random() < mutation_prob):
                a_path = self.offsprings[i].path
                l1 = random.randrange(len(a_path)-1)
                l2 = random.randrange(len(a_path)-1)
                while l1 == l2:
                    l2 = random.randrange(len(a_path)-1)
                t = a_path[l1]
                a_path[l1] = a_path[l2]
                a_path[l2] = t
        self.population = self.offsprings

# File writing functions
def get_file_data(file_path: str):
    try:
        with open(file_path, 'r') as file:
            data = file.readlines()
        return data 
    except Exception as err:
        print(err)
        return []
# Write to a file
def write_output(path, cities): 
    ans = ''
    for index in path:
        cords = cities[index].coords
        ans+= "{} {} {}\n".format(cords[0], cords[1],cords[2])
    ans+= "{} {} {}\n".format(cities[path[0]].coords[0], cities[path[0]].coords[1],cities[path[0]].coords[2])
    with open ("output.txt", 'w') as file:
        file.write(ans)
# Create cities list with indexes
def initialize_data(data, cities, index):
    for i in range(1, len(data)):
        data[i]= data[i].replace("\n", "")
        cities.append(City(list(map(int,data[i].split(" ")))))
        index.append(i-1)
# Initialize population and calculate the distnce for each
def initialPopulation(population, index):
    for i in range(POPSIZE):
        random.shuffle(index)
        population.append(Path(path=index.copy()))
run = Run()
history = {} # Keep track of minimum distance
random.seed(20)
file_data = get_file_data("input.txt")
initialize_data(file_data, run.cities, run.index)
initialPopulation(run.population, run.index)
history = {}
mini = run.calculateDistance()
run.calculateFitness()
history[mini] =  run.population[run.best_path]
for i in range(ITERATIONS):
    run.selectParents()
    run.crossOver()
    run.repairChildren()
    run.mutate()
    mini = run.calculateDistance()
    run.calculateFitness()
    history[mini] = run.population[run.best_path]
    # Handle cross prob and mutation prob over multiple iterations
    if(i > 50 and cross_prob > 0.9):
        cross_prob -= alpha_cross
    if(i > 400 and mutation_prob > 0.12):
        mutation_prob -= alpha_mut
write_output(history[min(list(history.keys()))].path, run.cities)