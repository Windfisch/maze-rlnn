from random import random
from itertools import product
from collections import defaultdict
from sys import argv

maze = [ [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 2] ]

start=(1,1)
theta = None
theta_1 = 0.00001
theta_2 = 0.01
gamma=0.9 # discount
epsilon = 0.1
epsilon_reduction = 1
alpha = 0.4
alpha_reduction = 0.9998
friendlyness = 0.7  # probability that the model actually performs the action we've requested.
                    # the model will perform a random other actions with probability of (1-friendlyness)/3.
                    # putting 0.25 here will make it just random.
frameskip = 99
visu = True


def cursor_up(n):
        print("\033[%dA" % n)

def args(argv):
    result=defaultdict(lambda:None)
    ss=None
    for s in argv[1:]+["-"]:
        if s[0]=='-':
            if ss!=None:
                result[ss]=True
            ss=s
        else:
            if ss!=None:
                result[ss]=s
                ss=None
            else:
                explode
    return result

arg=args(argv)
print(arg)

mode = None
if arg['-1'] or arg['--policy-evaluation']:
    mode = 1
elif arg['-2'] or arg['--q-learning']:
    mode = 2
else:
    print("Usage: %s MODE [OPTIONS]" % argv[0])
    print("       MODE:  -1 / --policy-evaluation or\n" +
          "              -2 / --q-learning\n" +
          "       OPTIONS: --theta NUM     # convergence threshold\n" +
          "                                # default: %f / %f for -1 / -2\n" % (theta_1, thetha_2) +
          "                --gamma NUM     # learning discount for value iteration\n" +
          "                                # default: %f\n" % gamma +
          "                --alpha NUM     # learning rate for q-learning\n" +
          "                                # default: %f\n" % alpha +
          "                --alphared NUM  # reduction of alpha per episode\n" +
          "                                # default: %f\n" % alpha_reduction +
          "                --friendly NUM  # friendlyness of the system (probability\n" +
          "                                  that the requested action is really done)\n" +
          "                                # default: %f\n" % friendlyness +
          "                --epsilon NUM   # value for the epsilon-policy used in q-learning\n" +
          "                                # default: %f\n" % epsilon +
          "                --epsred NUM    # reduction of epsilon per episode\n" +
          "                                # default: %f\n\n" % epsilon_reduction +
          "                --frameskip NUM # frameskip for visualisation\n" +
          "                                # default: %f\n" % frameskip +
          "                --quiet         # disable visualisation" +
          "                --file FILE     # output file for q learning")
    exit()


if arg['-q'] or arg['--quiet']:
    visu = False

if arg['--theta']:
    theta = theta_1 = theta_2 = float(arg['--theta'])

if arg['--gamma']:
    gamma = float(arg['--gamma'])

if arg['--epsilon']:
    epsilon = float(arg['--epsilon'])

if arg['--epsred']:
    epsilon_reduction = float(arg['--epsred'])

if arg['--alpha']:
    alpha = float(arg['--alpha'])

if arg['--alphared']:
    alpha_reduction = float(arg['--alphared'])

if arg['--friendly']:
    friendlyness = float(arg['--friendly'])

logfile = None
if arg['--file']:
    logfile = open(arg['--file'], "w")

NORTH=0
EAST=1
SOUTH=2
WEST=3

directions = [NORTH, EAST, SOUTH, WEST]
dir_coords = [(0,-1), (1,0), (0,1), (-1,0)]

def argmax(l):
    return max(range(len(l)), key=lambda i:l[i])

def merge_dicts(dicts):
    result = defaultdict(lambda:0.)
    for factor, d in dicts:
        for k in d:
            result[k] += factor * d[k]
    return result

def draw_randomly(d):
    c = 0.
    rnd = random()
    for k in d:
        c += d[k]
        if rnd < c:
            return k

def visualize(maze, P):
    n=0
    for y in range(len(maze)):
        line1=""
        line2=""
        line3=""
        for x in range(len(maze[0])):
            if maze[y][x] == 1:
                line1 += "@" * (2+7)
                line3 += "@" * (2+7)
                line2 += "@@@%03.1f@@@" % P[y][x]
            elif maze[y][x] == 2:
                line1 += "." * (2+7)
                line3 += "." * (2+7)
                line2 += ".%07.5f." % P[y][x]
            else:
                line1 += "'" + " " * (7) + " "
                line3 += " " + " " * (7) + " "
                line2 += " %07.5f " % P[y][x]
        print(line1)
        print(line3)
        print(line2)
        print(line3)
        print(line3)
        n+=5
    return n

def visualize2(maze, Q):
    n=0
    for y in range(len(maze)):
        line1=""
        line2=""
        line3=""
        line4=""
        line5=""
        for x in range(len(maze[0])):
            if maze[y][x] == 1:
                f = lambda s : s.replace(" ","@")
            elif maze[y][x] == 2:
                f = lambda s : s.replace(" ","+")
            else:
                f = lambda s : s

            maxdir = argmax(Q[y][x])
            line3 += f("'     " + ("^" if maxdir == NORTH else " ") + "     ")
            line5 += f("      " + ("v" if maxdir == SOUTH else " ") + "     ")
            line1 += f("    %04.2f    " % Q[y][x][NORTH])
            line2 += f("%s%04.2f  %04.2f%s" % ("<" if maxdir == WEST else " ",Q[y][x][WEST], Q[y][x][EAST], ">" if maxdir == EAST else " "))
            line4 += f("    %04.2f    " % Q[y][x][SOUTH])
        print(line3)
        print(line1)
        print(line2)
        print(line4)
        print(line5)
        n+=5
    return n

class World:
    def __init__(self, maze, pos):
        self.x,self.y = pos
        self.maze = maze
        self.xlen = len(maze[0])
        self.ylen = len(maze)

    def possible_next_states(self, s):
        # must return at least all possible states.
        # must only return valid states.
        x,y = s
        return filter(lambda s : s[0]>=0 and s[1]>=0 and s[0] < self.xlen and s[1] < self.ylen, [(x,y),(x+1,y),(x-1,y),(x,y-1),(x,y+1)])


    # definitely walks from (x,y) into direction.
    # returns the neighboring coordinate on success,
    # or the old one if there was a wall
    def walk(self, x,y, direction):
        dx,dy=dir_coords[direction]
        nx,ny = x+dx, y+dy

        if 0 <= nx and nx < self.xlen and \
           0 <= ny and ny < self.ylen and \
           self.maze[y][x] == 0 and \
           self.maze[ny][nx] != 1:
            return nx,ny
        else:
            return x,y


    # gives probabilities for new states, given
    # the command "direction".
    def action(self, x,y , direction):
        newstates = defaultdict(lambda:0.)
        for i in range(4):
            newstates[ self.walk(x,y, (direction+i)%4 ) ] += friendlyness if i == 0 else (1-friendlyness)/3. #[1.0,0.,0.,0.][i] # [0.7,0.1,0.1,0.1][i]
        return newstates

    def take_action(self, x,y, direction):
        newstates = self.action(x,y,direction)
        ss = draw_randomly(newstates)
        return self.R((x,y),ss, None), ss


    def P(self, s, ss, pi):
        return merge_dicts([(pi[d], self.action(s[0],s[1], d)) for d in directions])[ss]


    def R(self, s, ss, pi):
        if s!=ss and self.maze[ss[1]][ss[0]] == 2: # goal
            return 10.0
        else:
            return 0.

    def is_final(self,s):
        return self.maze[s[1]][s[0]] == 2

if mode == 1:  # policy evaluation
    theta = theta_1
    a = World(maze, start)

    V = [ [0.0] * a.xlen for i in range(a.ylen) ]
    pi = [ [ [0.25] * 4 for i in range(a.xlen) ] for j in range(a.ylen) ]

    i=0
    while True:
        i = i + 1
        delta = 0
        for x,y in product(range(a.xlen), range(a.ylen)):
            v = V[y][x]
            V[y][x] = sum( a.P((x,y),(xx,yy), pi[y][x]) * (  a.R((x,y),(xx,yy), pi[y][x]) + gamma * V[yy][xx]  ) for xx,yy in product(range(a.xlen), range(a.ylen)) )
            delta = max(delta, abs(v - V[y][x]))

        print("iteration %.3d, delta=%.7f"%(i,delta))
        n = 0
        if visu:
            n = visualize(maze,V)
        cursor_up(n+2)

        if (delta < theta):
            break
    print("finished after %d iterations" % i)
    visualize(maze,V)

elif True:  # q-learning
    theta = theta_2

    a = World(maze, start)
    Q = [ [ [0. for k in range(4)] for i in range(a.xlen) ] for j in range(a.ylen) ]

    i=0
    stopstate = -1
    total_reward = 0.
    for i in range(1000000):
        s = start
        maxdiff=0.
        for j in range(100):
            # epsilon-greedy
            greedy = argmax(Q[s[1]][s[0]])
            rnd = random()
            action = None
            if rnd < epsilon:
                action = ( greedy + int(1 + 3 * rnd / epsilon) ) % 4
            else:
                action = greedy

            r,ss = a.take_action(s[0],s[1], action)
            #print ((r,ss))
            diff = alpha * (r + gamma * max( [ Q[ss[1]][ss[0]][aa] for aa in directions ] ) - Q[s[1]][s[0]][action])
            Q[s[1]][s[0]][action] += diff
            maxdiff = max(abs(diff),maxdiff)
            total_reward += r
            s = ss
            if a.is_final(ss):
                break

        if (i % (frameskip+1) == 0):
            print("iteration %.3d, alpha=%.3e, epsilon=%.3e maxdiff=%.7f"%(i,alpha,epsilon,maxdiff))
            n = 0
            if visu:
                n = visualize2(maze,Q)
            cursor_up(n+2)

        if (logfile != None):
            print("%d\t%f" % (i, total_reward), file=logfile)

        # Wikipedia says on this: "When the problem is stochastic [which it is!],
        # the algorithm still converges under some technical conditions on the
        # learning rate, that require it to decrease to zero.
        # So let's sloooowly decrease our learning rate here. Otherwise it won't
        # converge, but instead oscillate plus/minus 0.5.
        # However, if we set the friendlyness of our system to 1.0, then it would
        # also converge without this learning rate reduction, because we have a
        # non-stochastic but a deterministic system now.
        alpha *= alpha_reduction
        epsilon *= epsilon_reduction

        
        # stop once we're below theta for at least 100 episodes. But not before we went above theta at least once.
        if maxdiff < theta:
            stopstate -= 1
            if stopstate == 0:
                break
        else:
            stopstate = 100
    
    print("finished after %.3d iterations, alpha=%.3e, epsilon=%.3e"%(i,alpha,epsilon))
    visualize2(maze,Q)
    if logfile != None:
        logfile.close()
