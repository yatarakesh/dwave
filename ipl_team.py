from dwave.system import DWaveSampler,EmbeddingComposite
from dimod import BinaryQuadraticModel
import random

#Assigning index for players
players = [i for i in range(50)]

#Assigning price for the players
price = []
for i in range(50):
    price.append(random.randint(50,100))   #Price in Lakhs


#Giving rating for the players
rating = []
for i in range(50):
    rating.append(random.randint(2,5))     #rating from 2 star to 5 star

#building a binary variable for each player  0:not selected , 1:selected
x = ['selection' for p in players]


bqm = BinaryQuadraticModel('BINARY')


#objective   -  Minimize the Cost
for p in players:
    bqm.add_variable(x[p],price[p])

#constraint 1  - Picking 5 batsmen out of 18 batsmen
c1 = [(x[i],1) for i in range(18)]
bqm.add_linear_equality_constraint(c1,constant = -5, lagrange_multiplier = 28)

#constraint 2 - picking 6 bowlers out of 18 bowlers
c2 = [(x[i],1) for i in range(18,36)]
bqm.add_linear_equality_constraint(c2,constant = -6, lagrange_multiplier = 28)

#constraint 3 - Picking 4 all-rounders out of 14 all-rounders
c3 = [(x[i],1) for i in range(36,50)]
bqm.add_linear_equality_constraint(c3,constant = -4, lagrange_multiplier = 28)

#constraint 4 - Overall rating should be greater than or equal to 50
c4 = [(x[i],rating[i]) for i in range(50)]
bqm.add_linear_inequality_constraint(c4, lb = 50, ub = 75, lagrange_multiplier = 13, label = 'rating')

#Giving the Dwave sampler our bqm model
sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample(bqm, num_reads=1000)
sample = sampleset.first.sample



# printing the result
overall_rating = 0
total_cost = 0

for p in players:
    print('p'+str(p))
    print("\t"+str(sample[x[p]]))
    total_cost = total_cost + price[p]*sample[x[p]]
    overall_rating = overall_rating + rating[p]*sample[x[p]]
    print()

print()
print("total cost:")
print(total_cost)
print()
print("total rating:")
print(overall_rating)





