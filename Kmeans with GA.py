#############################################################################
# Full Imports

import math
import random
import csv
"""
This is a pure Python implementation of the K-Means Clustering algorithmn. The
original can be found here:
http://pandoricweb.tumblr.com/post/8646701677/python-implementation-of-the-k-means-clustering
I have refactored the code and added comments to aid in readability.
After reading through this code you should understand clearly how K-means works.
If not, feel free to email me with questions and suggestions. (iandanforth at
gmail)
This script specifically avoids using numpy or other more obscure libraries. It
is meant to be *clear* not fast.
I have also added integration with the plot.ly plotting library. So you can see
the clusters found by this algorithm. To install run:
```
pip install plotly
```
This script uses an offline plotting mode and will store and open plots locally.
To store and share plots online sign up for a plotly API key at https://plot.ly.
"""

from Aux import *

class Individual (object):
    def __init__(self,x,k,genes):
        self.genes = genes
        if x!= None:
            for i in range (0,k):
                #random initial point
                point = x[randint(0,len(x)-1)]
                #gen = x
                for coord in point:
                    self.genes.append(coord)
            self.dim = len(x[0])
        else:
            self.dim = len(genes)/k

    #assign each point to a cluster
    def assign(self,x):
        output = []
        for point in x:
            distance = []
            for index in range (0,len(self.genes)/self.dim):
                distance.append(np.linalg.norm(np.array(point)-np.array(self.genes[index*self.dim:(index+1)*self.dim])))
            output.append(np.argmin(distance))
        return output

    #windexes of points that belong to a given cluster
    def elements (self,cluster,output):
        return np.where(np.array(output)==cluster)[0]

    #update clusters centroids based on assignments
    def update (self,x,output):
        for index in range (0,len(self.genes)/self.dim):
            xi = self.elements(index,output)
            for d in range(index*self.dim,(index+1)*self.dim):
                self.genes[d] = sum([x[item][d%self.dim] for item in xi])/len(xi) if len(xi)!=0 else self.genes[d]
    
    #intracluster distance = max d(i,j)
    def intracluster (self,x,output):
        intra = []
        for index in range(0,len(self.genes)/self.dim):
            xi = self.elements(index,output)
            dmax = 0
            for m,point1 in enumerate(xi):
                for point2 in xi[m+1:]:
                    d = np.linalg.norm(np.array(x[point1])-np.array(x[point2]))
                    if d > dmax:
                        dmax = d
            intra.append(dmax)
        return intra

    #intercluster distance for all clusters
    def intercluster (self):
        inter = []
        for index in range(0,len(self.genes)/self.dim):
            for j in range (index+1,len(self.genes)/self.dim):
                inter.append(np.linalg.norm(np.array(self.genes[index*self.dim:(index+1)*self.dim])-np.array(self.genes[j*self.dim:(j+1)*self.dim])))
        return inter

    def fitness (self,x):
        output = self.assign(x)
        self.update(x,output)
        return min(self.intercluster())/max(self.intracluster(x,self.assign(x)))

    def mutation (self,pmut):
        for g,gene in enumerate(self.genes):
            if uniform(0,1) <= pmut:
                delta = uniform(0,1)
                if uniform(0,1) <= 0.5:
                    self.genes[g] = gene - 2*delta*gene if gene!=0 else -2*delta
                else:
                    self.genes[g] = gene + 2*delta*gene if gene!=0 else 2*delta

def GAPopulationInit (npop,x,k):
    lists = [Individual(x,k,[]) for i in range (0,npop)]
    return lists

def Crossover (parent1, parent2,k):
    point = randint(1,len(parent1.genes)-2)
    return Individual(None,k,parent1.genes[:point]+parent2.genes[point:]), Individual(None,k,parent2.genes[:point]+parent1.genes[point:])

def RouletteWheel (pop,fit):
    sumf = sum(fit)
    prob = [(item+sum(fit[:index]))/sumf for index,item in enumerate(fit)]
    return pop[BinSearch(prob,uniform(0,1),0,len(prob)-1)]

def GeneticAlg (npop,k,pcros,pmut,maxit,arqStr):
    x,y = Inputs(arqStr)
    pop = GAPopulationInit(npop,x,k)
    fit = [indiv.fitness(x) for indiv in pop]
    verybest = [pop[np.argmax(fit)],max(fit)]
    for i in range(0,maxit):
        print "Iterasi %s" % (i+1) ,
        print "fitness : ", verybest[1]
        fit = [indiv.fitness(x) for indiv in pop]
        new = []
        while len(new) < len(pop):
            #selection
            parent1 = RouletteWheel(pop,fit)
            p = uniform(0,1)
            #genetic operators
            if p <= pcros:
                parent2 = RouletteWheel(pop,fit)
                while parent2 == parent1:
                    parent2 = RouletteWheel(pop,fit)
                child1, child2 = Crossover(parent1,parent2,k)
                new.append(child1)
                if len(new) < len(pop):
                    new.append(child2)
            else:
                child = deepcopy(parent1)
                child.mutation(pmut)
                new.append(child)
        pop = deepcopy(new)
        #elitism (but individual is kept outside population)
        if max(fit)>verybest[1]:
            verybest = [pop[np.argmax(fit)],max(fit)]
    print "\nFinal Fitness = %s" % verybest[1]
    #return best cluster
    p = verybest[0].genes[0:4]
    q = verybest[0].genes[4:8]
    r = verybest[0].genes[8:12]
    return initial_centroid(p,q,r)

def main():

    # How many points are in our dataset?
    num_points = 150

    # For each of those points how many dimensions do they have?
    # Note: Plotting will only work in two or three dimensions
    dimensions = 4

    # Bounds for the values of those points in each dimension
    lower = 0
    upper = 10

    # The K in k-means. How many clusters do we assume exist?
    num_clusters = 3

    # When do we say the optimization has 'converged' and stop updating clusters
    cutoff = 0.2

    # Generate some points to cluster
    #points = []
    with open ("dataset.csv", 'rb') as csvfile:
        lines = csv.reader(csvfile)
        points = [makeline(row) for row in lines]
    # points = [
    #     makeRandomPoint(dimensions, lower, upper) for i in xrange(num_points)
    # ]

    # Cluster those data!
    clusters = kmeans(points, num_clusters, cutoff)

    # Print our clusters
    for i, c in enumerate(clusters):
        cluster_n = 0
        for p in c.points:
            print " Cluster: ", i, "\t Point :", p
            cluster_n += 1
        print "Cluster", i, " = ", cluster_n

class Point(object):
    '''
    A point in n dimensional space
    '''
    def __init__(self, coords):
        '''
        coords - A list of values, one per dimension
        '''

        self.coords = coords
        self.n = len(coords)

    def __repr__(self):
        return str(self.coords)

class Cluster(object):
    '''
    A set of points and their centroid
    '''

    def __init__(self, points):
        '''
        points - A list of point objects
        '''

        if len(points) == 0:
            raise Exception("ERROR: empty cluster")

        # The points that belong to this cluster
        self.points = points

        # The dimensionality of the points in this cluster
        self.n = points[0].n

        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n:
                raise Exception("ERROR: inconsistent dimensions")

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)

    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.
        Note: Initially we expect centroids to shift around a lot and then
        gradually settle down.
        '''
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        print "New Centroid : ",self.centroid
        return shift

    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = []        

        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
        # print "coords centroid : ",centroid_coords
        return Point(centroid_coords)

def kmeans(points, k, cutoff):

    # Pick out k random points to use as our initial centroids
    #initial = random.sample(points, k)
    initial = GeneticAlg(int(150),int(3),float(0.85),float(0.01),int(3),"iris.txt")
    for elements in initial:
        print "initial cluster centroid : ", elements
    print "\n"
    # Create k clusters using those centroids
    # Note: Cluster takes lists, so we wrap each point in a list here.
    clusters = [Cluster([p]) for p in initial]

    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [[] for _ in clusters]
        clusterCount = len(clusters)

        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
            clusterIndex = 0

            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getDistance(p, clusters[i+1].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            # After finding the cluster the smallest distance away
            # set the point to belong to that cluster
            lists[clusterIndex].append(p)

        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # For each cluster ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)
        print "biggest_shift %s"%biggest_shift
        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print "Converged after %s iterations" % loopCounter
            break
    return clusters

def getDistance(a, b):
    '''
    Euclidean distance between two n-dimensional points.
    https://en.wikipedia.org/wiki/Euclidean_distance#n_dimensions
    Note: This can be very slow and does not scale well
    '''
    if a.n != b.n:
        raise Exception("ERROR: non comparable points")

    accumulatedDifference = 0.0
    for i in range(a.n):
        squareDifference = pow((a.coords[i]-b.coords[i]), 2)
        accumulatedDifference += squareDifference
    distance = math.sqrt(accumulatedDifference)

    return distance

def makeline(row):
    # for elements in row[:-1]:
    p = Point([float(elements) for elements in row[:-1]])
    return p

def initial_centroid(p,q,r):
    initial = []
    P = Point(p)
    Q = Point(q)
    R = Point(r)
    initial.append(P)
    initial.append(Q)
    initial.append(R)
    print 
    return initial

if __name__ == "__main__":
    main()
