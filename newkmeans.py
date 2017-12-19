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

    # Display clusters using plotly for 2d data
    if dimensions in [2, 3] and plotly:
        print "Plotting points, launching browser ..."
        plotClusters(clusters, dimensions)

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
        #print "points : ",points
        self.centroid = self.calculateCentroid()
        #print "old centroid:", old_centroid
        #print "selfcentroud:", self.centroid
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        # print "numpoints",numPoints
        # Get a list of all coordinates in this cluster
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
    initial = initial_centroid()

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
        print "%s\n"%loopCounter
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
            print "clusters i :",clusters[i]
            print "list i:",len(lists[0])
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
    print a
    print b
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

def initial_centroid():
    initial = []
    p = Point([5.649999999999999, 2.641176470588235, 4.047058823529412, 1.25])
    q = Point([5.005999999999999, 3.4180000000000006, 1.464, 0.2439999999999999])
    r = Point([6.5772727272727245, 2.990909090909091, 5.3484848484848495, 1.8954545454545448])
    initial.append(p)
    initial.append(q)
    initial.append(r)
    print initial
    return initial

if __name__ == "__main__":
    main()
