# The goal here is to take a given symmetric adjacency matrix or
# collectiion of edges and break it down into a sequence of edge
# collections with no node having more than one incoming edges. 
#
# The algorithm returns a two dimensional matrix that is has a row
# dimension equal to the number of nodes and column dimension equal to
# the number of separate edge sets. The matrix has the property that
# each column has no duplicated integers and each row contains all the
# neighbors of a node.
#
#

import random


size = 5
avg_n = 1
neighbors = [[] for x in range(size)]

for i in range(size):
    for j in range(min(size - 1, max(1, random.randrange(avg_n)))):
        while True:
            n = random.randrange(size)
            if( n in neighbors[i] or n == i):
                continue
            else:
                break                   
        neighbors[i].append(n)
        neighbors[n].append(i)

ordered_neighbors = [[] for x in range(size)]
neighbor_counts = [0 for x in range(size)]
paired = [False for x in range(size)]
print neighbors

finished = 0
for i in range(size): #for each round

    paired = [False for x in range(size)] #no one has a pair
    for j in range(size):
        if(not paired[j]):
            #this one is not yet paired

            #pessimistically assume it won't find a partner
            ordered_neighbors[j].append(-1)
            
            #we must find an unpaired neighbor
            for k in range(len(neighbors[j])):

                if(neighbors[j][k] in ordered_neighbors[j]):
                    continue #we already have paired with this one

                if(not paired[neighbors[j][k]]): 

                    #got an unpaired friend!

                    #The origin is j and j goes to its neighbor on the ith round            
                    ordered_neighbors[j][i] = neighbors[j][k]
                    #The other origin is j's neighbor, which goes to j on the ith round
                    ordered_neighbors[neighbors[j][k]].append(j)

                    #increment our counts
                    neighbor_counts[j] += 1
                    neighbor_counts[neighbors[j][k]] += 1
                    paired[neighbors[j][k]] = True
                    break
            #I either found out all my neighbors are paired, and thus I am sitting out
            #or I was successful
            paired[j] = True 

    #check if we're done
    finished = True
    for j in range(size):
        if(neighbor_counts[j] != len(neighbors[j])):
            finished = False
            break
        
    print ordered_neighbors

    if(finished):
        print "Process will require {} exchanges".format(i+1)
        break
    
