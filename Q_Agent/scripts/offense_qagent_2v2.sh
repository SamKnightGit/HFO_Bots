#!/bin/bash

# Should be run from the main HFO directory
# First argument is how many runs
# Second argument is how many games played per run

for i in {1..$1}
do	
	./bin/HFO --offense-agents=2 --defense-npcs=2 --offense-on-ball 1 --trials $2 --headless &> "./example/custom_agents/HFO_Bots/Q_Agent/scripts/offense_2v2_raw_data.txt" &

	sleep 15
	python ./example/custom_agents/HFO_Bots/Q_Agent/high_level_q_agent.py --playerIndex=1 --numEpisodes=$2 &> agent1.txt &
	sleep 5
	python ./example/custom_agents/HFO_Bots/Q_Agent/high_level_q_agent.py --playerIndex=2 --numEpisodes=$2 &> agent2.txt &
	sleep 10 
	
	# get relevant data from output 
	python ./example/custom_agents/HFO_Bots/Q_Agent/scripts/trim_data.py

	trap "kill -TERM -$$" SIGINT
done
