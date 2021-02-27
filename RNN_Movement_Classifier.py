import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, GRU
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


PATH = "parsed"
DATASETS = ['LTC','ZEC','ETH','BTC']
TARGET = 'ETH'
SEQ_LEN = 24
FUTURE_PERIOD_PREDICT = 3
EPOCHS = 5
BATCH_SIZE = 100
TRAIN_TEST_SPLIT = 0.20
SAMPLE_SPLIT = 0.10
LEARNING_RATE = 0.001
DECAY = 1e-6
LOSS = "binary_crossentropy"
OPTIMIZER = lambda lr, decay: tf.keras.optimizers.Adam()
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify(current, future):
	if float(future) > float(current):
		return 1
	else:
		return 0

def preprocess_df(dff):
	dff=dff.drop('future', 1)
	for col in dff.columns:
		#print(col)
		if col != "target":
			dff[col] = dff[col].pct_change()
			dff.dropna(inplace=True)
			dff[col] = preprocessing.scale(dff[col].values)
	dff.dropna(inplace=True)

	sequential_data = []
	prev_days = deque(maxlen=SEQ_LEN)

	for i in dff.values:
		prev_days.append([n for n in i[:-1]])
		if len(prev_days) == SEQ_LEN:
			sequential_data.append([np.array(prev_days), i[-1]])
	
	random.shuffle(sequential_data)
	buys = []
	sells = []
	for seq, target in sequential_data:
		if target == 0:
			sells.append([seq, target])
		elif target == 1:
			buys.append([seq, target])

	random.shuffle(buys)
	random.shuffle(sells)

	lower = min(len(buys),len(sells))
	buys = buys[:lower]
	sells = sells[:lower]

	sequential_data = buys+sells
	random.shuffle(sequential_data)

	X=[]
	y=[]
	for seq, target in sequential_data:
		X.append(seq)
		y.append(target)

	return np.array(X), y

main_df = pd.DataFrame()
for crypto in DATASETS:
	fullpath = f"{PATH}/{crypto}USD.csv"
	df = pd.read_csv(fullpath)
	df.rename(columns={colname:f"{crypto}_{colname}" for colname in df.columns[1:]},inplace=True)
	df.set_index('unix',inplace=True)
	main_df = df if len(main_df) == 0 else main_df.merge(df, on = 'unix')
	

main_df["future"] = main_df[f'{TARGET}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = [int(i) for i in list(map(classify, main_df[f'{TARGET}_close'], main_df['future']))]

times = sorted(main_df.index.values)
# current_df = main_df.iloc[:times[-24],:]
# main_df = main_df.iloc[times[-24]:,:]
# print(current_df.head())
split = times[-int(TRAIN_TEST_SPLIT*len(times))]
samplesplit = times[-int(SAMPLE_SPLIT*len(times))]
sample_df = main_df[(main_df.index >= samplesplit)]
main_df = main_df[(main_df.index < samplesplit)] 
validation_df = main_df[(main_df.index >= split)]
main_df = main_df[(main_df.index < split)]
train_x, train_y = preprocess_df(main_df)
test_x, test_y = preprocess_df(validation_df)
sample_x, sample_y = preprocess_df(sample_df)

print("Train/test split counts: ")
print(f"train_x features: {len(train_x)} | train_y targets: {len(train_y)}")
print(f"test_x features: {len(test_x)} | test_y targets: {len(test_y)}")
print(f"sample_x features: {len(sample_x)} | sample_y targets: {len(sample_y)}")

print('Balance:')
print(train_y.count(1),'|', train_y.count(0))
print(test_y.count(1),'|', test_y.count(0))
print(sample_y.count(1),'|', sample_y.count(0))

model = Sequential()
# model.add(LSTM(256,input_shape=(train_x.shape[1:]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# model.add(GRU(256,input_shape=(train_x.shape[1:]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

model.add(LSTM(128,input_shape=(train_x.shape[1:]),return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="tanh"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))

opt = OPTIMIZER(lr=LEARNING_RATE, decay=DECAY)
model.compile(loss=LOSS,
				optimizer=opt,
				metrics=['accuracy','mean_squared_error'])

#tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

#filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"
#checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))

history = model.fit(train_x, np.asarray(train_y),
					batch_size=BATCH_SIZE,
					epochs=EPOCHS,
					validation_data=(test_x,np.asarray(test_y)))#,callbacks=[tensorboard, checkpoint])


predictions = model.predict(sample_x)

predicted_y = []
for prediction in predictions:
	predicted_y.append(float(prediction))

notsure_right = 0
sure_right = 0
notsure_wrong = 0
sure_wrong = 0
total = 0
for p_y, s_y in zip(predicted_y, sample_y):
	if (0.60 <= float(p_y) or 0.40 >= float(p_y)) and int(round(p_y)) == s_y:
		sure_right += 1
	elif (0.60 > float(p_y) and 0.40 < float(p_y)) and int(round(p_y)) == s_y:
		notsure_right += 1
	elif (0.60 <= float(p_y) or 0.40 >= float(p_y)) and int(round(p_y)) != s_y:
		sure_wrong += 1
	elif (0.60 > float(p_y) and 0.40 < float(p_y)) and int(round(p_y)) != s_y:
		notsure_wrong += 1
	total += 1

print('\n<------------------>')
print("At least 70% sure: ")
print('Right',sure_right)
print('Wrong',sure_wrong)
print('Accuracy',round(sure_right/(sure_right+sure_wrong),2))
print('\n<------------------>')
print("Below 70% sure: ")
print('Right',notsure_right)
print('Wrong',notsure_wrong)
print('Accuracy',round(notsure_right/(notsure_right+notsure_wrong),2))
print('\n<------------------>')
print('Total: ')
print('Accuracy',round((sure_right+notsure_right)/total,2))
print(total)

#model.evaluate(sample_x, np.asarray(predicted_y))