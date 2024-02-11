import MyClass as mc

# ************** read dataframe ****************
files = mc.read_file('Data_Train.xlsx, Data_Test.xlsx')
train, test = files.read_excel()
# **********************************************

print(train['Reviews'])
# *********** drop column 'Synopsis' *********
train.drop('Synopsis', axis = 1, inplace=True)
test.drop('Synopsis', axis = 1, inplace = True)
# *******************************************


# ******************* get values in row **********************
value = mc.getValue()
train['Reviews'] = value.getValueReview(train['Reviews'])
value.getValueEdition(train['Edition'])
train['Ratings'] = value.getValueRating(train['Ratings'])

test['Reviews'] = value.getValueReview(test['Reviews'])
value.getValueEdition(test['Edition'])
test['Ratings'] = value.getValueRating(test['Ratings'])
# ***********************************************************


# *************** data to numerical for 'BookCategory' and 'Genre' ********************
trainDataToNum = mc.data_to_num(train, 'BookCategory')
trainDataToNum.convert()

testDataToNum = mc.data_to_num(test, 'BookCategory')
testDataToNum.convert()

trainDataToNum = mc.data_to_num(train, 'Genre')
trainDataToNum.convert()

testDataToNum = mc.data_to_num(test, 'Genre')
testDataToNum.convert()

trainDataToNum = mc.data_to_num(train, 'Title')
trainDataToNum.convert()

testDataToNum = mc.data_to_num(test, 'Title')
testDataToNum.convert()

trainDataToNum = mc.data_to_num(train, 'Author')
trainDataToNum.convert()

testDataToNum = mc.data_to_num(test, 'Author')
testDataToNum.convert()

trainDataToNum = mc.data_to_num(train, 'Edition')
trainDataToNum.convert()

testDataToNum = mc.data_to_num(test, 'Edition')
testDataToNum.convert()
#*************************************************************************


# **************** set features and targets ****************
x_train = train[[ i for i in train.columns if i != 'Price']]
x_test = test
y_train = train['Price']
# ***********************************************************
print(train['Reviews'])

# ***************** neural network regression ******************
max_iter = 1000
hidden_layer = 50
alpha = 0.0001
random = 27
ntr = mc.neural_network_regression(x_train, y_train, x_test)
print(ntr.predict(max_iter, hidden_layer, alpha, random))

