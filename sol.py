from random import random
from itertools import product
from collections import defaultdict
from sys import argv
from fann2 import libfann

maze = [ [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 2] ]

start=(1,1)
theta = 0.01
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
if arg['-h']:
    print("Usage: %s [OPTIONS]" % argv[0])
    print("       OPTIONS: --theta NUM     # convergence threshold\n" +
          "                                # default: %f\n" % theta +
          "                --gamma NUM     # learning discount\n" +
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
          "                --qfunc TYPE    # type of the Q function's representation\n" +
          "                                  arr / array -> plain standard array\n" +
          "                                  nn          -> neural network representation\n" +
          "                                  default: array" +
          "                --frameskip NUM # frameskip for visualisation\n" +
          "                                # default: %f\n" % frameskip +
          "                --quiet         # disable visualisation\n" +
          "                --file FILE     # output file for q learning")
    exit()


if arg['-q'] or arg['--quiet']:
    visu = False

if arg['--theta']:
    theta = float(arg['--theta'])

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

def draw_randomly(d):
    c = 0.
    rnd = random()
    for k in d:
        c += d[k]
        if rnd < c:
            return k

def visualize(maze, Q):
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

            Qev = Q.eval(x,y)
            maxdir = argmax(Qev)
            line3 += f("'       " + ("^" if maxdir == NORTH else " ") + "       ")
            line5 += f("        " + ("v" if maxdir == SOUTH else " ") + "       ")
            line1 += f("     %06.2f     " % Qev[NORTH])
            line2 += f("%s%06.2f  %06.2f%s" % ("<" if maxdir == WEST else " ",Qev[WEST], Qev[EAST], ">" if maxdir == EAST else " "))
            line4 += f("     %06.2f     " % Qev[SOUTH])
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


    def R(self, s, ss, pi):
        if s!=ss and self.maze[ss[1]][ss[0]] == 2: # goal
            return 10.0
        else:
            return 0.

    def is_final(self,s):
        return self.maze[s[1]][s[0]] == 2


# abstracts the Q-array. semantics of .eval(x,y) is `Q[y][x]`. semantics of .change((x,y),ac,diff) is `Q[y][x][ac]+=diff`
class QArray:
    def __init__(self):
        self.Q = [ [ [0. for k in range(4)] for i in range(a.xlen) ] for j in range(a.ylen) ]

    def eval(self,x,y = None):
        if y==None: x,y = x

        return self.Q[y][x]
    
    def change(self, s, action, diff):
        self.Q[s[1]][s[0]][action] += diff


# implements the Q function not through an array, but through a neuronal network instead.
class QNN:
    def __init__(self):
        connection_rate = 1
        num_input = 2
        hidden = (40,40)
        num_output = 4
        learning_rate = 0.7

        self.NN = libfann.neural_net()
        self.NN.create_sparse_array(connection_rate, (num_input,)+hidden+(num_output,))
        self.NN.set_learning_rate(learning_rate)
        #self.NN.set_activation_function_input(libfann.SIGMOID_SYMMETRIC_STEPWISE)
        self.NN.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
        self.NN.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
        #self.NN.set_activation_function_output(libfann.LINEAR)
    
    def eval(self,x,y = None):
        if y==None: x,y = x

        return [x*10. for x in self.NN.run([x,y])]
    
    def change(self, s, action, diff):
        oldval = self.eval(s)
        newval = list(oldval) # copy list
        newval[action] += diff

        self.NN.train(list(s), [x/10. for x in newval])

a = World(maze, start)

Q = None
if arg['--qfunc'] == "nn":
    Q = QNN()
else:
    Q = QArray()

i=0
stopstate = -1
total_reward = 0.
for i in range(1000000):
    s = start
    maxdiff=0.
    for j in range(100):
        # epsilon-greedy
        greedy = argmax(Q.eval(s))
        rnd = random()
        action = None
        if rnd < epsilon:
            action = ( greedy + int(1 + 3 * rnd / epsilon) ) % 4
        else:
            action = greedy

        r,ss = a.take_action(s[0],s[1], action)
        #print ((r,ss))
        diff = alpha * (r + gamma * max( [ Q.eval(ss)[aa] for aa in directions ] ) - Q.eval(s)[action])
        Q.change(s,action,diff)
        maxdiff = max(abs(diff),maxdiff)
        total_reward += r
        s = ss
        if a.is_final(ss):
            break

    if (i % (frameskip+1) == 0):
        print("iteration %.3d, alpha=%.3e, epsilon=%.3e maxdiff=%.7f"%(i,alpha,epsilon,maxdiff))
        n = 0
        if visu:
            n = visualize(maze,Q)
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
        stopstate = 1000

print("finished after %.3d iterations, alpha=%.3e, epsilon=%.3e"%(i,alpha,epsilon))
visualize(maze,Q)
if logfile != None:
    logfile.close()
