#!/usr/bin/env python3.5
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
relevant_data = os.path.join(curr_dir, 'offense_2v2_relevant_data.txt')
raw_data = os.path.join(curr_dir, 'offense_2v2_raw_data.txt')

with open(relevant_data, 'a+') as data_file:
	with open(raw_data, 'r') as raw_file:
		relevant_data = False
		for line in raw_file:
			if relevant_data and "[start.py]" in line:
				relevant_data = False
			if "TotalFrames" in line:
				relevant_data = True
			if relevant_data:
				data_file.write(line)

				
		
