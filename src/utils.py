from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

import tensorflow as tf
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical


def load_california_housing_data():
	# load the california housing data from csv
	train_file = '/content/sample_data/california_housing_train.csv'
	test_file = '/content/sample_data/california_housing_test.csv'

	train_combined = pd.read_csv(train_file)
	test = pd.read_csv(test_file)

	# split the data in validation and test (from test.csv)
	train, val = train_test_split(train_combined, test_size=0.25, random_state=seed)  # random state to ensure reproducible split across multiple function call

	feature_names = list(train_combined.columns)
	print(feature_names)

	# assign the target variables
	target = 'median_house_value'

	# extract the target label in all sets
	train_labels_df= train[target]
	val_labels_df = val[target]
	test_labels_df = test[target]

	# extract the data from all sets 
	train_data_df = train.drop(columns=target, axis=1)
	val_data_df = val.drop(columns=target, axis=1)
	test_data_df = test.drop(columns=target, axis=1)

	train_data_unnormalized = train_data_df.to_numpy()
	train_labels_unnormalized = train_labels_df.to_numpy()

	val_data_unnormalized = val_data_df.to_numpy()
	val_labels_unnormalized = val_labels_df.to_numpy()

	test_data_unnormalized = test_data_df.to_numpy()
	test_labels_unnormalized = test_labels_df.to_numpy()

	# normalize the data using minmax 
	minmax = MinMaxScaler() 

	train_data = minmax.fit_transform(train_data_unnormalized)
	train_label_temp = np.expand_dims(train_labels_unnormalized, axis=1)
	train_labels = minmax.fit_transform(train_label_temp)

	val_data = minmax.fit_transform(val_data_unnormalized)
	val_label_temp = np.expand_dims(val_labels_unnormalized, axis=1)
	val_labels = minmax.fit_transform(val_label_temp)

	test_data = minmax.fit_transform(test_data_unnormalized)
	test_label_temp = np.expand_dims(test_labels_unnormalized, axis=1)
	test_labels = minmax.fit_transform(test_label_temp)


	print('Training data shape \n', train_data.shape)
	print('Training labels shape \n', train_labels.shape)
	#print('Training data \n ', train_data)
	#print('Training labels \n ', train_labels)

	print('Validation data shape \n ',val_data.shape)
	print('Validation labels shape \n ', val_labels.shape)
	#print('Validation data \n ', val_data)
	#print('Validation labels \n ', val_labels)

	print('Test data shape \n ', test_data.shape)
	print('Test labels shape \n ', test_labels.shape)

	#print('Test data \n ', test_data)
	#print('Test labels \n ', test_labels)# load the california housing data from csv

	return train_data, train_labels, val_data, val_labels, test_data, test_labels, feature_names


def load_cifar(num_classes, 
               val_split, 
               rotation_range , 
               width_shift_range, 
               height_shift_range, 
               shear_range,
	           zoom_range,
               horizontal_flip,
	           vertical_flip,
	           rescale ,
               train_batch_size,
               val_batch_size,
               test_batch_size):

	num_classes = num_classes
	val_split=val_split
	rotation_range=rotation_range
	width_shift_range=width_shift_range
	height_shift_range=height_shift_range
	shear_range=shear_range
	zoom_range=zoom_range
	horizontal_flip=horizontal_flip
	vertical_flip=vertical_flip
	rescale=rescale
	train_batch_size=train_batch_size
	val_batch_size=val_batch_size
	test_batch_size=test_batch_size

	# load the dataset and set the augmentation configuration 


	(x_train, y_train), (x_test, y_test) = load_data()
	
	# normalize pixel values from 0 to 1 
	x_train = x_train / 255.0
	x_test = x_test / 255.0

	# convert labels to one hot encoding
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test, num_classes)


	#x_test = x_test/255.
	#y_train = to_categorical(y_train)
	#y_test = to_categorical(y_test) 

	number_of_holdout_examples = int(x_test.shape[0]*val_split)

	# use after the rescaling issue has been figured (rescale does not seem to change the pixel value) 
	#train_datagen = ImageDataGenerator(rotation_range=rotation_range,
	#	                               width_shift_range=width_shift_range,
	#	                               height_shift_range=height_shift_range,
	#	                               shear_range=shear_range,
	#	                               zoom_range=zoom_range,
	#	                               horizontal_flip=horizontal_flip,
	#	                               vertical_flip=vertical_flip)
	#	                               rescale=rescale)
	#	                               validation_split=val_split)

	#validation_datagen = ImageDataGenerator(rescale=rescale)
	#test_datagen = ImageDataGenerator(rescale=rescale)

	train_datagen = ImageDataGenerator(rotation_range=rotation_range,
		                               width_shift_range=width_shift_range,
		                               height_shift_range=height_shift_range,
		                               shear_range=shear_range,
		                               zoom_range=zoom_range,
		                               horizontal_flip=horizontal_flip,
		                               vertical_flip=vertical_flip)

	validation_datagen = ImageDataGenerator()
	test_datagen = ImageDataGenerator() 		                               
    
    
	train_generator = train_datagen.flow(x_train, y_train, batch_size=train_batch_size)
	validation_generator = validation_datagen.flow(x_test[0:number_of_holdout_examples], y_test[0:number_of_holdout_examples], batch_size=val_batch_size)
	test_generator = test_datagen.flow(x_test[number_of_holdout_examples:], y_test[number_of_holdout_examples:], batch_size=test_batch_size)

	print('number of examples in train generator ', train_generator.n)
	print('number of examples in val generator ', validation_generator.n)
	print('number of examples in test generator ', test_generator.n)

	return (x_train, y_train), (x_test, y_test), train_generator, validation_generator, test_generator


# function to visualize the ground truth with the predicted value and (corridor of uncertainty)

def plot_gt_vs_pred(ground_truth, prediction_mean, prediction_std, path, indices_to_be_plotted, uncert_name):
	plt.figure(figsize=(30, 4))
	plt.plot(range(ground_truth.shape[0]),  ground_truth, color='k', label='ground truth', marker='o')
	plt.plot(range(ground_truth.shape[0]), prediction_mean, color='r', label='prediction', marker='o')

	y_pred_mean = prediction_mean.reshape((-1,))
	y_pred_std = prediction_std.reshape((-1,))
	y_pred_up_1 = y_pred_mean + y_pred_std
	y_pred_down_1 = y_pred_mean - y_pred_std

	plt.fill_between(range(ground_truth.shape[0]), y_pred_down_1, y_pred_up_1, color=(0, 0, 0.9, 0.7), label='corridor of uncertainty ($\pm$ 1 $\sigma$) ', alpha=0.5)
	#plt.plot(range(ground_truth.shape[0]), y_pred_mean, '.', color=(0, 0.9, 0.0, 0.8), markersize=0.2, label='Mean')

	#plt.set_title('{}\nInterval Score: {:.2f}'.format(key, score))
	#plt.set_ylim([-20.0, 20.0])

	#plt.axvline(x=-4.0, color='black', linestyle='dashed')
	#plt.axvline(x= 4.0, color='black', linestyle='dashed')
	#plt.get_xaxis().set_ticks([])
	#plt.get_yaxis().set_ticks([])    

	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	#plt.legend()
	plt.grid()
	plt.xticks(range(len(indices_to_be_plotted)), indices_to_be_plotted, rotation=45)
	plt.xlabel('Input sample #')
	plt.ylabel('Target Variable (normalized)')
	plt.title('Ground Truth and '+uncert_name+' model prediction')
	plt.savefig(path+'.pdf')
	plt.savefig(path+'.png')
	plt.show()


# code for horizontal bar chart 

def plot_explanation(grads_plot, feature_names, sample_number=None, err=None, save_file_path=None, combination_title=None):
	plt.figure(figsize=(15, 10))
	#plt.barh(pos, vals, color=colors) #this code works well but does not have legend or text in int

	colors = ['C1' if x > 0 else 'C0' for x in list(grads_plot)] # originally negative=red and positive=green (coloring scheme)
	colors_set = set(colors)
	#print('colors_set before ', colors_set)
	colors_set = ['positive' if c == 'C1' else 'negative' for c in colors_set]
	#print('colors_set after ', colors_set)
	exp = list(grads_plot)
	pos = np.arange(len(exp)) + .5
	#print('pos values for plot are ', pos)

	vals = [float(x) for x in exp]
	vals_str = [str(round(val, 3)) for val in vals]
	#print('vals_str ', vals_str)

	#colors = ['r', 'g', 'b']
	labels = colors_set 
	legend_colors = list(set(colors))
	#print(legend_colors)
	handles = [plt.Rectangle((0,0),1,1, color=legend_colors[label]) for label in range(len(labels))]
	#print(handles)co

	vals = [np.abs(num) if num  == 0 else num for num in vals] # removing the sign from 0 vals
	#print('vals ', vals)

	#max_horizontal_value_to_plot_text_box = np.max(vals)

	for i, v in enumerate(vals):
		if err is None:
			# in case the text box needs to be moved along the bar then switch 0 to v
			plt.text(v, i+0.5, str(round(v, 3)), Bbox = dict(facecolor = 'grey', alpha =0.2)) # https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/

		elif err is not None:
			# in case of the combined explanation, the text box should contain the mean+-std as well
			plt.text(v, i+0.7, str(round(v, 3))+'$\pm$'+str(round(err[i], 3)), Bbox = dict(facecolor = 'grey', alpha =0.2))

	axx = plt.barh([i for i in pos], vals, xerr=err, align='center', color=colors) # this code appropriate legend

	plt.grid(alpha=0.5)
	plt.ylabel('feature names')
	plt.xlabel('feature coefficient')
	plt.yticks(pos, feature_names[:-1])

	if err is not None:
		error = plt.plot([], label='corridor of uncertainty ($\pm \sigma$)', linewidth=3, linestyle='-', color='k')
		plt.title(f'combined explanation , prediction : {combination_title[0]} $\pm$ {combination_title[1]} , GT : {combination_title[2]}')
		main_bars = [plt.Rectangle((0,0),1,1, color=legend_colors[label]) for label in range(len(labels))]
		#first_legend = plt.legend(handles=error)
		#second_legend = plt.legend(handles=main_bar)

		handles = error + main_bars #https://stackoverflow.com/questions/28732845/combine-two-pyplot-patches-for-legend       
		error_label = ['corridor of uncertainty ($\pm \sigma$)']
		main_bars_label = colors_set

		labels = error_label + main_bars_label
		plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

	else:
		handles=[plt.Rectangle((0,0),1,1, color=legend_colors[label]) for label in range(len(labels))]
		plt.title(f'explanation for sample_number {sample_number}')
		plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5)) # https://stackoverflow.com/questions/57340415/matplotlib-bar-plot-add-legend-from-categories-dataframe-column

	plt.subplots_adjust()
	plt.savefig(save_file_path+'.pdf')
	plt.savefig(save_file_path+'.png')



