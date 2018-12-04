#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.contrib import rnn

def load_data():
    print("Start to load data")
    col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
    
    data = pd.read_csv("kddcup.data.corrected", names = col_names)
    
    #change all label to 1 or 0, for each 1 equals to normal, 0 equals attack
    data.loc[data['label']!='normal.','label']=0
    data.loc[data['label']=='normal.','label']=1
    
    '''
    Applying long short-term memory recurrent neural networks to intrusion detection
    https://www.researchgate.net/publication/279770740_Applying_long_short-term_memory_recurrent_neural_networks_to_intrusion_detection
    only 8 features are actually useful for taining
    '''
    selected_column=['service','src_bytes', 'dst_host_diff_srv_rate','dst_host_rerror_rate', 'dst_bytes', 'hot', 'num_failed_logins', 'dst_host_srv_count','label']
    selected_data = data[selected_column]
    #use 20 percent of data to train and 80 percent to test
    percent_train = 0.2
    len_train = int(len(selected_data)*percent_train)
    
    #get word count and instances from 'service' column
    serdict = {}
    word_counter = 0
    for record in selected_data["service"]:
        if record in serdict:
            serdict[record] = serdict[record]+1 
        else:
            serdict[record] = 1
        word_counter = word_counter + 1
    
    #encode service column using the each instance's proportion
    for word in serdict:
        selected_data.loc[selected_data['service']==word,'service']=serdict[word]/word_counter
    

    selected_features = ['service','src_bytes','dst_host_diff_srv_rate','dst_host_rerror_rate','dst_bytes','hot','num_failed_logins','dst_host_srv_count']
    
    #fit each input value using Standard Scaler
    features = np.array(selected_data[selected_features])
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    features = features.reshape([-1,features.shape[1],1])
    
    x_train = features[:len_train,:]
    x_test = features[len_train,:]
    
    #encode label to one hot label
    labels = y=np.array(selected_data['label']).reshape([-1,1])
    one_hot_label = np.zeros([labels.shape[0],2])
    for i in range(0, y.shape[0]):
        if y[i,0]==1:
            one_hot_label[i,1]=1
        else:
            one_hot_label[i,0]=1
    
    y_train = one_hot_label[:len_train,:]
    y_test = one_hot_label[len_train,:]
    print("Data loading finished!")
    return x_train,y_train,x_test,y_test,len_train;


# In[4]:

def LSTM(X, weight, bias):
    X = tf.unstack(X, 8, 1)
    output_list=[]

    for i in range(block_num):
        with tf.variable_scope("block_"+str(i), reuse=tf.AUTO_REUSE):
            lstm_cel = rnn.LSTMCell(hidden_unit*cell_num, forget_bias=1.0,use_peepholes=False)
            outputs, states = rnn.static_rnn(lstm_cel, X, dtype=tf.float32)
            output_list.append(outputs[-1])

    output=tf.concat(output_list,1)
    return tf.matmul(output, weight['out']) + bias['out']


# In[5]:


x_train,y_train,x_test,y_test,len_train = load_data()

# initailize all hyper parameters
learning_rate = 0.001
batch_size = 128
process_display = 200
EPOCH=10

#hyper parameter for specific requirement
block_num = 2
cell_num = 2
class_num = 2
timesteps = 8
hidden_unit = 15

X = tf.placeholder("float", [None, timesteps, 1])
Y = tf.placeholder("float", [None, class_num])

# initialize weights and bias
weight = {
    'out': tf.Variable(tf.random_normal([hidden_unit*cell_num*block_num, 2]))
}
bias = {
    'out': tf.Variable(tf.random_normal([2]))
}

logits = LSTM(X, weight, bias)
prediction = tf.nn.softmax(logits)
loss_operator = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_operator = optimizer.minimize(loss_operator)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

# from https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
# to find the way of creating variable TP TN FP FN
argmax_prediction = tf.argmax(prediction, 1)
argmax_label = tf.argmax(Y, 1)
TP = tf.count_nonzero(argmax_prediction * argmax_label, dtype=tf.float32)
TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_label - 1), dtype=tf.float32)
FP = tf.count_nonzero(argmax_prediction * (argmax_label - 1), dtype=tf.float32)
FN = tf.count_nonzero((argmax_prediction - 1) * argmax_label, dtype=tf.float32)

Total = TP+TN+FP+FN
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1= 2*Recall * Precision / (Recall + Precision)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[6]:

#generate shuffled index for minibatch
def minibatch_shuffle(batch_size, index):
    arange = np.arange(index)
    np.random.shuffle(arange)
    count = 0
    while count < index:
        batch = arange[count:count+batch_size]
        count = count + batch_size
        yield np.array(batch)

# In[7]:


init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    for epoch in range(EPOCH):
        print ("Initializing epoch:",epoch + 1)
        
        step_count=0
        for index in minibatch_shuffle(batch_size, x_train.shape[0]):
            x_batch, y_batch = x_train[index],y_train[index]
            sess.run(training_operator, feed_dict={X: x_batch, Y: y_batch})
            #for each process_display count show the program process on 'loss' 'accuracy' 'f1 score' and 'recall value'
            if step_count % process_display == 0:
                loss, acc,f1,recall = sess.run([loss_operator, accuracy,F1,Recall], feed_dict={X: x_batch, Y: y_batch})
                print("Progress Update |"
                        + " Loss: {:.4f}".format(loss) 
                        + ", Accuracy: {:.3f}".format(acc) 
                        + ", F1 score: {:.3f}".format(f1)
                        + ", Recall value: {:.3f}".format(recall)
                    )
            step_count+=1

print("Finished!")
# In[ ]:




