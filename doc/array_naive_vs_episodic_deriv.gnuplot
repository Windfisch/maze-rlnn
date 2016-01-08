set xrange [0: 1000]

plot 'derivs/old.log' with lines lt rgb 'red' title 'QArray, old textbook strategy, n=100',\
     'derivs/new.log' with lines lt rgb 'green' title 'QArray, new reversed strategy, n=100',\
     'derivs/nn2.log' using ($1, $2*20) with lines lt rgb 'blue' title 'QNN, n=30',\
     'derivs/nn.log' using ($1, $2*20) with lines lt rgb 'light-blue' title 'QNN, friendlyness=1, n=1'
pause -1
