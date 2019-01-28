#!/usr/bin/env python3.5
import os
import time 
import subprocess

curr_dir = os.path.dirname(os.path.abspath(__file__))
relevant_data = os.path.join(curr_dir, 'offense_2v2_relevant_data.txt')
raw_data = os.path.join(curr_dir, 'offense_2v2_raw_data.txt')

with open(relevant_data, 'a+') as data_file:
	with open(raw_data, 'r') as raw_file:
		training_in_progress = True
		while training_in_progress:
			final_line = subprocess.check_output(['tail', '-1', raw_data])
			print(final_line)
			if "[start.py]" in final_line:
				training_in_progress = False
			time.sleep(5)
		
		results = subprocess.check_output(['tail', '-7', raw_data])
		data_file.write(results)

				
		
