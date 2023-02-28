import os
import numpy as np

import torch
import torch.nn as nn

import physionet.data_utils as utils
from torch.distributions import uniform

from torch.utils.data import DataLoader

from physionet.physionet import PhysioNet, variable_time_collate_fn, get_data_min_max,get_data_mean_std,get_data_features

from sklearn import model_selection
import random
import pdb

#####################################################################################################
def parse_datasets(args, device):
	

	def basic_collate_fn(batch, time_steps, args = args, device = device, data_type = "train"):
		batch = torch.stack(batch)
		data_dict = {
			"data": batch, 
			"time_steps": time_steps}

		data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
		return data_dict

	dataset_name = args.dataset

	##################################################################
	##################################################################
	# Physionet dataset

	if dataset_name == "physionet":

		train_dataset_obj = PhysioNet('physionet', train=True, 
										quantization = args.quantization,
										download=True, n_samples = 10000, 
										device = device)
		# Use custom collate_fn to combine samples with arbitrary time observations.
		# Returns the dataset along with mask and time steps
		test_dataset_obj = PhysioNet('physionet', train=False, 
										quantization = args.quantization,
										download=True, n_samples = 10000, 
										device = device)


		# Combine and shuffle samples from physionet Train and physionet Test
		total_dataset = train_dataset_obj[:len(train_dataset_obj)]

		if not args.classif:
			# Concatenate samples from original Train and Test sets
			# Only 'training' physionet samples are have labels. Therefore, if we do classifiction task, we don't need physionet 'test' samples.
			total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]

		if args.data_mode == 0:

			# Shuffle and split
			#results in 400 testing samples

			train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.95, 
				random_state = 42, shuffle = True)
			resume = 'physionet/PhysioNet/processed/statistics_{0}.pt'.format(args.quantization)


			val_data = train_data[-400:]
			train_data = train_data[:-400]
			if os.path.isfile(resume):
				statistics = torch.load(resume,map_location = device)
				data_min, data_max = statistics["mean"],statistics["std"]
			else:
				data_min, data_max = get_data_mean_std(train_data)
				state = {"mean": data_min,"std":data_max}
				torch.save(state,resume)
		elif args.data_mode ==4:

			resume = 'physionet/PhysioNet/processed/data_mode_{0}_statistics_{1}.pt'.format(args.data_mode,args.quantization)

			if not os.path.isfile(resume):

				real_total_dataset = []
				temp_total_dataset = []
				time_diff = 12.0
				sample_ratio = 0.95

				#remove "too late observations" from dataset

				for y in range(len(total_dataset)):
					record_id, tt, vals, mask, labels = total_dataset[y]

					idx = torch.where(((tt <= (tt[-1]- time_diff)).float() + (tt == tt[-1]).float()) > 0)

					new_tt = tt[idx]
					new_vals = vals[idx]
					new_mask = mask[idx]

					#i.e. we an observation before the prediction time
					if len(idx[0]) > 1:
						temp_total_dataset.append([record_id, new_tt, new_vals, new_mask, labels ])
						real_total_dataset.append([record_id, tt, vals, mask, labels ])


				train_data, test_data = model_selection.train_test_split(real_total_dataset, train_size= sample_ratio, 
					random_state = 42, shuffle = True)

				temp_data, _ = model_selection.train_test_split(temp_total_dataset, train_size= sample_ratio, 
					random_state = 42, shuffle = True)

				temp_data = temp_data[:-400]
				val_data = train_data[-400:]
				train_data = train_data[:-400]
				data_min, data_max = get_data_mean_std(temp_data)
				state = {"mean": data_min,"std":data_max, "train_data": train_data, "test_data":test_data,"val_data":val_data}
				torch.save(state,resume)
			else:
				statistics = torch.load(resume,map_location = device)
				data_min, data_max = statistics["mean"],statistics["std"]
				train_data, test_data = statistics["train_data"], statistics["test_data"]
				val_data = statistics["val_data"]

		else:
			if args.classif:
				print("PROBLEM WITH STORED DATA")
				pdb.set_trace()

			if args.data_mode == 1:
				time_diff = 6.0
				sample_ratio = 0.95
			elif args.data_mode == 2:
				time_diff = 12.0
				sample_ratio = 0.95
			elif args.data_mode == 3:
				time_diff = 24.0
				sample_ratio = 0.95
			else:
				print("unexpected data_mode", args.data_mode)
				exit()

			#resume = 'physionet/PhysioNet/processed/data_mode_{0}_statistics_{1}.pt'.format(args.data_mode,args.quantization)

			real_total_dataset = []
			#remove "too late observations" from dataset

			for y in range(len(total_dataset)):
				record_id, tt, vals, mask, labels = total_dataset[y]

				idx = torch.where(((tt <= (tt[-1]- time_diff)).float() + (tt == tt[-1]).float()) > 0)

				new_tt = tt[idx]
				new_vals = vals[idx]
				new_mask = mask[idx]

				#i.e. we an observation before the prediction time
				if len(idx[0]) > 1:
					real_total_dataset.append([record_id, new_tt, new_vals, new_mask, labels ])

			train_data = real_total_dataset[0:-800]
			val_data = real_total_dataset[-800:-400]
			test_data = real_total_dataset[-400:]

			data_min, data_max = get_data_mean_std(train_data)
			state = {"mean": data_min,"std":data_max, "train_data": train_data, "test_data":test_data,"val_data":val_data}
			#torch.save(state,resume)
				
		record_id, tt, vals, mask, labels = train_data[0]


		input_dim = vals.size(-1) - 6
		input_init = 6

		batch_size = min(len(train_data), args.batch_size)
		attr_names = train_dataset_obj.params

		if args.dataratio != 0 and False:

			counts = [0,0]
			for i in range(len(train_data)):
				if int(train_data[i][-1]) == 1:
					counts[1] += args.dataratio
				else:
					counts[0] += 1

			normalizer = counts[0] + counts[1]
			class_weights = [counts[i] / normalizer for i in range(len(counts))]
			class_weights.reverse()		
			weights = [class_weights[int(train_data[i][-1])] for i in range(len(train_data))]
			#pdb.set_trace()
			weights = torch.tensor(weights).to(device)
			sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_data))

			train_dataloader = DataLoader(train_data, sampler = sampler,batch_size= batch_size, 
				collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
					data_min = data_min, data_max = data_max))
		else:
			train_dataloader = DataLoader(train_data, shuffle = True,batch_size= batch_size, 
				collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
					data_min = data_min, data_max = data_max))

		test_dataloader = DataLoader(test_data, batch_size = 1, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max))

		val_dataloader = DataLoader(val_data, batch_size = 1, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max))

	data_objects = {"dataset_obj": train_dataset_obj, 
					"train_dataloader": train_dataloader, 
					"test_dataloader": test_dataloader,
					"val_dataloader":val_dataloader,
					"input_dim": input_dim,
					"input_init": input_init,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"attr": attr_names, #optional
					"classif_per_tp": False, #optional
					"n_labels": 1} #optional

	return data_objects

def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()
def add_mask(data_dict):
	data = data_dict["observed_data"]
	mask = data_dict["observed_mask"]

	if mask is None:
		mask = torch.ones_like(data).to(get_device(data))

	data_dict["observed_mask"] = mask
	return data_dict

