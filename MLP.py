#https://androidkt.com/k-fold-cross-validation-with-tensorflow-keras/

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense , Dropout
import tensorflow as tf
import numpy
from extractdata import data #the class for extracting data

# from keras.optimizers import SGD
from tensorflow.keras import layers
from matplotlib import pyplot as plt

def create_model():

    #The model is linear and fully connected
    #A dense layer is a layer in neural network that’s fully connected.
    #In other words, all the neurons in one layer are connected to all other neurons in the next layer.
    #ReLU stands for rectified linear unit.
    #ReLU is defined mathematically as F(x) = max(0,x).
    #In other words, the output is x, if x is greater than 0, and the output is 0 if x is 0 or negative.
    inputs = tf.keras.Input(shape=(28,))
    x = layers.Dense(28, activation='linear')(inputs)
    x = layers.Dense(14, activation='linear')(x)
    x = layers.Dense(7, activation='relu')(x)
    outputs = layers.Dense(3,activation='softmax')(x)
    #‘softmax‘ activation in order to predict the probability for each class.

    # inputs = tf.keras.Input(shape=(28,))
    # x = layers.Dense(28, activation='tanh')(inputs)
    # x = layers.Dense(14, activation='tanh')(x)
    # x = layers.Dense(7, activation='tanh')(x)
    # outputs = layers.Dense(3,activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    #learning process
    #loss: The function to minimize during optimization
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses
    #optimizer: This object specifies the training procedure
    #metrics: Used to monitor training
        #https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model

#Get the data
d = data()
d.get_mlp_data()
x = d.MergedTable.to_numpy()#[22467 rows x 28 columns]
y = d.labels.to_numpy()#[22467 rows x 1 column]

# #for our last test we split the dataset for training and validation data
# #22467 rows of data in Match Table
# #15727 + 6740 // split the dataset to 70% training and 30% testing
# x = d.MergedTable.head(15727).to_numpy()
# x_new = d.MergedTable.tail(6740).to_numpy()
# y = d.labels.head(15727).to_numpy()
# y_new = d.labels.tail(6740).to_numpy()

callbacks = [
      # Interrupt training if `val_loss` stops improving for over N epochs
      tf.keras.callbacks.EarlyStopping(patience=350, monitor='val_loss')
    ]

n_folds = 10
B365_money,BW_money,IW_money,LB_money = 0,0,0,0
for train_index , test_index in KFold(n_splits=n_folds,random_state=None, shuffle=True).split(x):
    x_train , x_test = x[train_index] , x[test_index]
    y_train , y_test = y[train_index] , y[test_index]

    model=create_model()

    #epochs: Training is structured into epochs.
        #An epoch is one iteration over the entire input data (this is done in smaller batches).
    model.fit(x_train, y_train,epochs=500, callbacks=callbacks,
          # We pass some validation for
          # monitoring validation loss and metrics
          # at the end of each epoch
          validation_data=(x_test, y_test))

    print('Model evaluation ' , model.evaluate(x_test,y_test))
    #evaluate : Returns the loss value & metrics values for the model in test mode.


    #predict : Generates output predictions for the input samples.
    y_Pred = model.predict(x_test)
    y_Pred = numpy.argmax(y_Pred,axis=1)
    print("How many Predictions are correct : ",numpy.sum(y_Pred == y_test),"/",len(y_Pred))
    print("\n Most Predicted Actual Result : ",numpy.bincount(y_Pred[y_Pred == y_test]).argmax())

    winning_odds = [0,0,0,0]
    for i in range(len(y_Pred)):
        #making an array with the odds of correctly predicted games
        if y_Pred[i] == y_test[i]:
            if y_Pred[i] == 0:#assos
                # winning_odds = numpy.vstack((winning_odds,x_test[i,0] ) )
                winning_odds = numpy.vstack((winning_odds,x_test[i,numpy.r_[0,3,6,9]] ) )
            elif y_Pred[i] == 1:#x
                winning_odds = numpy.vstack((winning_odds,x_test[i,numpy.r_[1,4,7,10]] ) )
            elif y_Pred[i] == 2:#diplo
                winning_odds = numpy.vstack((winning_odds,x_test[i,numpy.r_[2,5,8,11]] ) )

    #Count how much money we earn/lose from each booker
    B365_money += numpy.sum(winning_odds[:,0])-numpy.sum(y_Pred != y_test)
    BW_money += numpy.sum(winning_odds[:,1])-numpy.sum(y_Pred != y_test)
    IW_money += numpy.sum(winning_odds[:,2])-numpy.sum(y_Pred != y_test)
    LB_money += numpy.sum(winning_odds[:,3])-numpy.sum(y_Pred != y_test)

    # plt.figure(1)
    # plt.subplot(121)
    # plt.scatter(y_test,y_Pred)
    # plt.xlabel("Actual Values")
    # plt.ylabel("Predicted Values")
    # plt.show()

# #predict : Generates output predictions for the input samples.
# y_Pred = model.predict(x_new)
# y_Pred = numpy.argmax(y_Pred,axis=1)
# print("How many Predictions are correct : ",numpy.sum(y_Pred == y_new),"/",len(y_Pred))
# print("\n Most Predicted Actual Result : ",numpy.bincount(y_Pred[y_Pred == y_new]).argmax())
#
# winning_odds = [0,0,0,0]
# for i in range(len(y_Pred)):
#     #making an array with the odds of correctly predicted games
#     if y_Pred[i] == y_new[i]:
#         if y_Pred[i] == 0:#assos
#             # winning_odds = numpy.vstack((winning_odds,x_test[i,0] ) )
#             winning_odds = numpy.vstack((winning_odds,x_new[i,numpy.r_[0,3,6,9]] ) )
#         elif y_Pred[i] == 1:#x
#             winning_odds = numpy.vstack((winning_odds,x_new[i,numpy.r_[1,4,7,10]] ) )
#         elif y_Pred[i] == 2:#diplo
#             winning_odds = numpy.vstack((winning_odds,x_new[i,numpy.r_[2,5,8,11]] ) )
# #Print how much money we earn/lose from each booker
# print("With BetSize = 1 (1 Euro per bet) : \n From B365 we have : ",numpy.sum(winning_odds[:,0])-numpy.sum(y_Pred != y_test),
# "\n From BW : ",numpy.sum(winning_odds[:,1])-numpy.sum(y_Pred != y_test),
# "\n From IW : ",numpy.sum(winning_odds[:,2])-numpy.sum(y_Pred != y_test),
# "\n From LB : ",numpy.sum(winning_odds[:,3])-numpy.sum(y_Pred != y_test) )

#Print how much money we earn/lose from each booker
print(B365_money,BW_money,IW_money,LB_money)
#Print the model layout
# tf.keras.utils.plot_model(model, 'my_first_model_with_shape_info1.png', show_shapes=True)
