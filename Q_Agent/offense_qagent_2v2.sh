#!/bin/bash

# Should be run from the main HFO directory

./bin/HFO --offense-agents=2 --defense-npcs=2 --offense-on-ball 1 --trials $1 --headless &

sleep 15
python ./example/custom_agents/HFO_Bots/Q_Agent/high_level_q_agent.py --playerIndex=1 --numEpisodes=$1 &> agent1.txt &
sleep 5
python ./example/custom_agents/HFO_Bots/Q_Agent/high_level_q_agent.py --playerIndex=2 --numEpisodes=$1 &> agent2.txt &
sleep 5

trap "kill -TERM -$$" SIGINT