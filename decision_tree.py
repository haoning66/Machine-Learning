import math
import random
from copy import deepcopy



class decision_tree(object):
    def __init__(self,value,parent=None):
        super(decision_tree, self).__init__()
        self.value = value
        self.parent = parent
        self.children = {}

    def add_child(self, condition, value, child=None):
        if child is None:
            child = decision_tree(value)
        child.parent = self
        self.children.update({condition:value})
        return child

    def print_tree(self):
        children = self.children
        print(self.value)
        print(self.children)
        for child in children.values():
            if isinstance(child,decision_tree):
                print(self.value+' child')
                child.print_tree()




def getdata(filepath):
    file = open(filepath,'r')
    data = file.readlines()
    if '\n' in data:
        data.remove('\n')
    datas = []
    for item in data:
        datas.append(item.strip().split(','))
    return datas


def importance(attribute_no,examples):
    classes=[]
    for item in examples:
        if not item[-1] in classes:
            classes.append(item[-1])

    def calculate_entropy(x,y):
        if x == 0.0 and y != 0.0:
            if y==1.0:
                return 0.0
            else:
                return -1.0 * (y * math.log2(y) + (1.0 - y) * (math.log2(1.0 - y)))
        elif y == 0.0 and x != 0.0:
            if x==1.0:
                return 0.0
            else:
                return -1.0 * (x * math.log2(x) + (1.0 - x) * (math.log2(1.0 - x)))
        elif x == 0.0 and y == 0.0:
            return 0.0
        elif (1.0-(x+y)) == 0.0:
            return -1.0 * (x * math.log2(x) + y * math.log2(y))
        else:
            return -1.0 * (x * math.log2(x) + y * math.log2(y) + (1.0 - (x + y)) * (math.log2(1.0 - (x + y))))
    attribute_value = []
    reminder_value=0.0
    for item1 in examples:
        if not item1[attribute_no] in attribute_value:
            attribute_value.append(item1[attribute_no])
    for item2 in attribute_value:
        x=0
        y=0
        z=0
        for item3 in examples:
            if item3[attribute_no]==item2:
                z+=1
                if item3[-1]==classes[0]:
                    x+=1
                elif item3[-1]==classes[1]:
                    y+=1
        reminder_value += (z/len(examples))*calculate_entropy((x/z),(y/z))
    return round((1.0-reminder_value),3)


def classification_same(examples):
    example = []
    for item in examples:
        example.append(item[-1])
    classification_set = set(example)
    if len(classification_set) == 1:
        return True
    else:
        return False


def plurality_value(examples):
    classes=[]
    for item1 in examples:
        classes.append(item1[-1])
    temp = 0
    for item2 in classes:
        if classes.count(item2) > temp:
            max_class = item2
            temp = classes.count(item2)
    return max_class


def decision_tree_learning(examples, attributes, parent_examples,attributes_g,examples_g):
    if len(examples) == 0:
        return plurality_value(parent_examples)
    elif classification_same(examples):
        return examples[0][-1]
    elif len(attributes) == 0:
        return plurality_value(examples)
    else:
        a_importance = {}
        vk = []
        exs = []
        for attribute in attributes:
            a_importance.update({attribute:importance(attributes_g.index(attribute),examples)})
        A = max(a_importance,key=a_importance.get)
        tree = decision_tree(A)
        for item in examples_g:
            if item[attributes_g.index(A)] not in vk:
                vk.append(item[attributes_g.index(A)])
        for item1 in vk:
            attributes_c = deepcopy(attributes)
            for item2 in examples:
                if item2[attributes_g.index(A)]==item1:
                    exs.append(item2)
            attributes_c.remove(A)
            subtree = decision_tree_learning(exs,attributes_c,examples,attributes_g,examples_g)
            tree.add_child(item1,subtree)
            exs.clear()
        return tree


def test_tree(example,tree,attributes):
    for condition in tree.children.keys():
         if example[attributes.index(tree.value)] == condition:
             if not isinstance(tree.children[condition],decision_tree):
                 if example[-1] == tree.children[condition]:
                     return True
                 else:
                     return False
             else:
                 return test_tree(example, tree.children[condition], attributes)



if __name__ == '__main__':
    example1 = getdata('AIMA_Restaurant-data.txt')
    example2 = getdata('iris.data.discrete.txt')
    example4 = getdata('car.txt')
    attributes1 = ['Alternate','Bar','Fri/Sat','Hungry','Patrons','Price','Raining','Reservation','Type','WaitEstimate']
    attributes2 = ['sepal length','sepal width','petal length','petal width']
    attributes4 = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    random.shuffle(example2)
    random.shuffle(example4)
    example1_train = example1[0:10]
    example1_test = example1[2:]
    example2_train = example2[0:120]
    example2_test = example2[120:]
    example4_train = example4[0:1555]
    example4_test = example4[1555:]
    tree1 = decision_tree_learning(example1_train, attributes1, None, attributes1, example1_train)
    tree2 = decision_tree_learning(example2_train, attributes2, None, attributes2, example2_train)
    tree4 = decision_tree_learning(example4_train, attributes4, None, attributes4, example4_train)
    right1 = 0
    for item1 in example1_test:
        if test_tree(item1,tree1,attributes1):
            right1+=1
    correctness = "%.2f%%" % ((right1/(len(example1_test))) * 100)
    print('AIMA restaurant dataset, Precision:'+correctness)
    right2=0
    for item2 in example2_test:
        if test_tree(item2,tree2,attributes2):
            right2+=1
    correctness = "%.2f%%" % ((right2/(len(example2_test))) * 100)
    print('Iris dataset, Precision:'+correctness)
    right3=0
    for item3 in example4_test:
        if test_tree(item3,tree4,attributes4):
            right3+=1
    correctness = "%.2f%%" % ((right3/(len(example4_test))) * 100)
    print('Car evaluation dataset, Correctness:'+correctness)









