import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm

import utils
import utils.workspace as ws
from utils.output_xyz import output_xyz
from utils.eval_metric import IOU
from networks.model import Model
import utils.dataloader as dataloader
from utils.Logger import Logger
import utils.utils as utils


	
class LearningRateSchedule:
	def get_learning_rate(self, epoch):
		pass


class ConstantLearningRateSchedule(LearningRateSchedule):
	def __init__(self, value):
		self.value = value

	def get_learning_rate(self, epoch):
		return self.value


class StepLearningRateSchedule(LearningRateSchedule):
	def __init__(self, initial, interval, factor):
		self.initial = initial
		self.interval = interval
		self.factor = factor

	def get_learning_rate(self, epoch):

		return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
	def __init__(self, initial, warmed_up, length):
		self.initial = initial
		self.warmed_up = warmed_up
		self.length = length

	def get_learning_rate(self, epoch):
		if epoch > self.length:
			return self.warmed_up
		return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

	schedule_specs = specs["LearningRateSchedule"]
	print(schedule_specs)

	schedules = []

	for schedule_specs in schedule_specs:
		if schedule_specs["Type"] == "Step":
			
			schedules.append(
				StepLearningRateSchedule(
					schedule_specs["Initial"],
					schedule_specs["Interval"],
					schedule_specs["Factor"],
				)
			)
		elif schedule_specs["Type"] == "Warmup":
			schedules.append(
				WarmupLearningRateSchedule(
					schedule_specs["Initial"],
					schedule_specs["Final"],
					schedule_specs["Length"],
				)
			)
		elif schedule_specs["Type"] == "Constant":
			schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

		else:
			raise Exception(
				'no known learning rate schedule of type "{}"'.format(
					schedule_specs["Type"]
				)
			)

	return schedules

def get_spec_with_default(specs, key, default):
	try:
		return specs[key]
	except KeyError:
		return default

def init_seeds(seed=0):
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main_function(experiment_directory, continue_from, input_object):
	torch.cuda.empty_cache()
	init_seeds()
	out_dir = 'checkpoints/'+experiment_directory+'/'
	experiment_directory = out_dir
	logger = logging.getLogger()
	handler = logging.FileHandler(out_dir+'logfile.log')
	logger.addHandler(handler) 
	logger.debug("running " + experiment_directory)
	specs = ws.load_experiment_specifications('configs')

	logging.info("Experiment description: \n" + specs["Description"])

	arch = __import__("networks." + specs["NetworkArch"], fromlist=["PolyNet", "Decoder"])

	checkpoints = list(
		range(
			specs["SnapshotFrequency"],
			specs["NumEpochs"] + 1,
			specs["SnapshotFrequency"],
		)
	)
	
	for checkpoint in specs["AdditionalSnapshots"]:
		checkpoints.append(checkpoint)
	checkpoints.sort()
	print(checkpoints)
	lr_schedules = get_learning_rate_schedules(specs)

		
	def save_checkpoints(epoch):

		ws.save_model_parameters(experiment_directory, str(epoch) + ".pth", operation, optimizer_operation, epoch)
	
			
	def signal_handler(sig, frame):
		logging.info("Stopping early...")
		sys.exit(0)

	def adjust_learning_rate(lr_schedules, optimizer, epoch):		

		for i, param_group in enumerate(optimizer.param_groups):
			param_group["lr"] = lr_schedules[0].get_learning_rate(epoch)
			print(param_group["lr"])

	start_time = time.time()
	signal.signal(signal.SIGINT, signal_handler)



	operation = Model(ef_dim = 256)
	operation = operation.cuda()
	logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))


	num_epochs = specs["NumEpochs"]
	mse = nn.MSELoss(reduction = 'mean')

	
	

	if args.input_object != None:
		occ_dataset_train = dataloader.DataLoader(
		test_flag=True, inpt = args.input_object
		)	
		occ_dataset_test = dataloader.DataLoader(
		test_flag=True, inpt = args.input_object
		)



	train_loader = data_utils.DataLoader(
		occ_dataset_train,
		batch_size=1,
		shuffle=False,
		num_workers=6
	)

	test_loader = data_utils.DataLoader(
		occ_dataset_test,
		batch_size=1,
		shuffle=False,
		num_workers=6
	)


	num_scenes = len(occ_dataset_train)
	logging.info("There are {} shapes".format(num_scenes))

	logging.debug(operation)
	optimizer_operation = torch.optim.Adam(
		[
			{
				"params": (operation.parameters()),
				"lr": lr_schedules[0].get_learning_rate(0),
				"betas": (0.5, 0.999),
			},
		]
	)



	start_epoch = 0
	if continue_from is not None: 
		
		logging.info('continuing from "{}"'.format(continue_from))
		load = torch.load(experiment_directory+'operation_checkpoint_'+str(continue_from)+'.pth')
		operation.load_state_dict(load["operation_state_dict"])
		optimizer_operation.load_state_dict(load["optimizer_state_dict"])
		model_epoch = load["epoch"]		
		start_epoch = model_epoch + 1
		logging.debug("loaded")
		logging.info("starting from epoch {}".format(start_epoch))

	operation.train()



	
	last_epoch_time = 0
	best_iou = torch.zeros(len(train_loader))




	BEST_IOU = 0
	for epoch in range(start_epoch, start_epoch + num_epochs):
		

		adjust_learning_rate(lr_schedules, optimizer_operation, epoch - start_epoch)

		TOTAL_LOSS = 0
		for inds_inout, all_points, all_points_high, dimension, shape_names  in tqdm(train_loader):
			
			dimension = dimension.cuda()
			inds_inout = inds_inout.cuda()
			all_points = all_points.cuda()
		
			current = -torch.ones_like(inds_inout)
	
			total_loss, outputs = operation(current, all_points, inds_inout, 100, out_dir, torch.mean(best_iou.detach()), dimension, epoch)	
			
			
			if not math.isnan(total_loss):
				TOTAL_LOSS += total_loss.detach()/len(train_loader)

			
			
			optimizer_operation.zero_grad()	
			total_loss.backward()
			optimizer_operation.step()		     


			del total_loss
			del outputs




		if (epoch-start_epoch+1) in checkpoints:
			save_checkpoints(epoch)		
		
		if (epoch+1) % 1 == 0:
			#operation.eval()
			IOU_total = []
			with torch.no_grad():
				inds_inout, all_points, all_points_high, dimension, shape_names = next(iter(test_loader))
				
				dimension = dimension.cuda()
				inds_inout = inds_inout.cuda()
				all_points = all_points.cuda()
				all_points_high = all_points_high.cuda()

				
				current = -torch.ones_like(inds_inout)

				_, outputs = operation(current, all_points, inds_inout, 100, out_dir, best_iou[shape_names], dimension, epoch)	
				iou = IOU(outputs, inds_inout)
				IOU_total.append(iou)
				if best_iou[shape_names]<iou:
					best_iou[shape_names]=iou


				if shape_names % 10 ==0:
					output_xyz(all_points[0,outputs[0]<0], out_dir+str(shape_names)+'_output.ply')
					output_xyz(all_points[0,inds_inout[0]<0], out_dir+str(shape_names)+'gt.ply')

				average_best_iou = sum(IOU_total)/len(IOU_total)
			logging.debug('Average IOU:\t{:.6f}'.format(average_best_iou))


		if BEST_IOU < average_best_iou:

			BEST_IOU = average_best_iou
			model_path = out_dir+"operation_checkpoint_"+str(epoch)+".pth"
			torch.save({
				'epoch': epoch,
				'operation_state_dict': operation.state_dict(),
				'optimizer_state_dict': optimizer_operation.state_dict(),
			}, model_path)	
			model_path2 = out_dir+"best_checkpoint"+".pth"
			torch.save({
				'epoch': epoch,
				'operation_state_dict': operation.state_dict(),
				'optimizer_state_dict': optimizer_operation.state_dict(),
			}, model_path2)	
			logging.debug('BEST IOU:\t{:.6f}'.format(BEST_IOU))

			seconds_elapsed = time.time() - start_time
			ava_epoch_time = (seconds_elapsed - last_epoch_time)/10
			last_epoch_time = seconds_elapsed
	
			
		logging.debug("epoch = {}/{} , \
			total_loss={:.6f}".format(epoch, num_epochs+start_epoch, TOTAL_LOSS))



if __name__ == "__main__":

	import argparse

	arg_parser = argparse.ArgumentParser(description="Train a Network")
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True,
		help="The experiment directory. This directory should include "
		+ "experiment specifications in 'specs.json', and logging will be "
		+ "done in this directory as well.",
	)
	arg_parser.add_argument(
		"--continue",
		"-c",
		dest="continue_from",
		help="A snapshot to continue from. This can be an integer corresponding to an epochal snapshot.",
	)

	arg_parser.add_argument(
		"--gpu",
		"-g",
		dest="gpu",
		required=True,
		help="gpu id",
	)

	arg_parser.add_argument(
		"--input_object",
		"-i",
		dest = "input_object",
		required = True, 
		help = "The object name"
		
	)


	utils.add_common_args(arg_parser)

	args = arg_parser.parse_args()

	utils.configure_logging(args)
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

	if args.input_object == 'all':
		args.input_object = None

	directory_path = "checkpoints/" + str(args.experiment_directory)

	# Check if the directory already exists
	if not os.path.exists(directory_path):
		# If it doesn't exist, create it
		os.makedirs(directory_path)
		print(f"Directory '{directory_path}' created.")
	else:
		print(f"Directory '{directory_path}' already exists.")
	#os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)

	main_function(args.experiment_directory, args.continue_from, args.input_object)
