import monkdata as m
import dtree as dt
import drawtree_qt5 as dtqt
import matplotlib.pyplot as plt
import numpy as np



# Task 1: Compute the entropy of the data sets
entropy_monk1 = dt.entropy(m.monk1)
entropy_monk2 = dt.entropy(m.monk2)
entropy_monk3 = dt.entropy(m.monk3)

entropy_training_monk = [entropy_monk1, entropy_monk2, entropy_monk3]
print('-------- Task 1 --------')
print('\n')
print('Entropy computation')
for i in range(len(entropy_training_monk)):
    print('E[MONK-' + str(i+1) + '] = ' + str(entropy_training_monk[i]) )

# Task3: Information Gain
print('\n')
print('-------- Task 3 --------')
print('Information Gain computation')
print('\n')
#  Dataset 1
information_gain_monk1 = []
for i in range(len(m.attributes)):
    information_gain_monk1.append(dt.averageGain(m.monk1, m.attributes[i]))
print('Information Gain in Monk1: ')
print(information_gain_monk1)
print('Split should occur at atrribute: ' + str(1+ information_gain_monk1.index(max(information_gain_monk1))))
#  Dataset 2
information_gain_monk2 = []
for i in range(len(m.attributes)):
    information_gain_monk2.append(dt.averageGain(m.monk2, m.attributes[i]))
print('Information Gain in Monk2: ')
print(information_gain_monk2)
print('Split should occur at atrribute: ' + str(1+ information_gain_monk2.index(max(information_gain_monk2))))
#  Dataset 3
information_gain_monk3 = []
for i in range(len(m.attributes)):
    information_gain_monk3.append(dt.averageGain(m.monk3, m.attributes[i]))
print('Information Gain in Monk3: ')
print(information_gain_monk3)
print('Split should occur at atrribute: ' + str(1 +information_gain_monk3.index(max(information_gain_monk3))))

# Task4: Entropy of the subsets
# We take a closer look at the entropy of the subsets after the split
entropy_after_split = dt.entropy_subsets(m.monk1, m.attributes[4])
print('\n')
print('-------- Task 3 --------')
print('\n')
print('Entropy of the subsets after the split')
print(entropy_after_split)

subsets_list_level1, information_gain_level1 = dt.create_subsets_compute_information_gain(m.monk1, m.attributes, 4, True)
print('\n')
print('-------- Task 5 --------')
print('\n')
print('Information Gain for Level 2')
for i in range(len(information_gain_level1)):
    print('Information Gain for node a5 == ' + str(i + 1))
    print(information_gain_level1['a5 == ' + str(i + 1)])

most_common_values_lvl2 = {}
splits = [0, 3, 5, 1]
for i in range(len(subsets_list_level1)):
    if i == 0:
        most_common_values_lvl2['B1LN1'] = dt.mostCommon(subsets_list_level1[0])
        continue
    subsets_list_level2, information_gain_level2 = dt.create_subsets_compute_information_gain(subsets_list_level1[i], m.attributes, splits[i], False)
    for j in range(len(subsets_list_level2)):
        most_common_values_lvl2['B' +str(i+1) + 'LN' + str(j+1)] = dt.mostCommon(subsets_list_level2[j])
print(most_common_values_lvl2.values())

# Build tree 1 and compare the results
tree1 = dt.buildTree(m.monk1, m.attributes, maxdepth=2)
# dtqt.drawTree(built_tree)
print('Tree 1:')
print(tree1)

# Build all trees 
tree1 = dt.buildTree(m.monk1, m.attributes)
tree2 = dt.buildTree(m.monk2, m.attributes)
tree3 = dt.buildTree(m.monk3, m.attributes)

print('Training Data:')
# Compare the performance of the trees with the training data
print('Error rate Training Data (Tree 1):' +  str(1 - dt.check(tree1, m.monk1)))
print('Error rate Training Data (Tree 2):' +  str(1 - dt.check(tree2, m.monk2)))
print('Error rate Training Data (Tree 3):' +  str(1 - dt.check(tree3, m.monk3)))
print('Test Data:')
# Compare the performance of the trees with the test data
print('Error rate Test Data (Tree 1):' +  str(1 - dt.check(tree1, m.monk1test)))
print('Error rate Test Data (Tree 2):' +  str(1 - dt.check(tree2, m.monk2test)))
print('Error rate Test Data (Tree 3):' +  str(1 - dt.check(tree3, m.monk3test)))

# Task7: Pruning

# Obtain test and validation data sets
fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

mean_error_test_set1, std_error_test_set1 =  dt.reduced_error_pruning(m.monk1, m.monk1test, m.attributes, fractions ,300)
mean_error_test_set3, std_error_test_set3 =  dt.reduced_error_pruning(m.monk3, m.monk3test, m.attributes, fractions ,300)

fig, axes = plt.subplots(2, 1, sharex=True)  # Create two subplots sharing the same x-axis


# Subplot 1
axes[0].plot(fractions, mean_error_test_set1, 'b-o', label='MONK1')
axes[0].plot(fractions, mean_error_test_set3, 'g-x', label='MONK3')
axes[0].set_title('Mean of the error')
axes[0].set_ylabel('Mean error [-]')
axes[0].legend(loc="upper right")

# Subplot 2
axes[1].plot(fractions, std_error_test_set1, 'b-o', label='MONK1')
axes[1].plot(fractions, std_error_test_set3, 'g-x', label='MONK3')
axes[1].set_title('Standard deviation of the error')
axes[1].set_xlabel('Fractions [-]')
axes[1].set_ylabel('Standard Deviation [-]')
axes[1].legend(loc="upper right")

plt.show()

# Lowest mean error after pruning
me_ap_monk1 = np.min(mean_error_test_set1)
me_ap_monk3 = np.min(mean_error_test_set3)

print('Mean after post pruning for MONK-1: ' + str(me_ap_monk1))
print('Mean after post pruning for MONK-3: ' + str(me_ap_monk3))




