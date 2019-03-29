#!/usr/bin/env python3.5

import subprocess
import click
import os
import typing
import time
from datetime import datetime
from tqdm import tqdm, trange
from plotting import plot_data

creation_time = datetime.isoformat(datetime.today())

qtable_dir = os.path.dirname(os.path.realpath(__file__)) + '/../qtables/'  # type: str
qtable_dir += creation_time

output_dir = os.path.dirname(os.path.realpath(__file__)) + '/../output/'   # type: str
output_dir += creation_time

logging_dir = os.path.dirname(os.path.realpath(__file__)) + '/../logs/'
logging_dir += creation_time

TRAIN_Q_AGENT_PATH = './example/custom_agents/HFO_Bots/Q_Agent/q_agent_training.py'
TEST_Q_AGENT_PATH = './example/custom_agents/HFO_Bots/Q_Agent/q_agent_testing.py'


@click.command()
@click.option('--num_agents', '-na', default=2)
@click.option('--num_opponents', '-no', default=2)
@click.option('--num_iterations', '-n', default=4,
              help="The number of training iterations to run the bots on.")
@click.option('--trials_per_iteration', '-t', default=50000,
              help="The number of trials (games) of HFO in each iteration.")
@click.option('--port', '-p', default=6000,
              help="Port which main HFO server will run on.")
@click.option('--epsilon_start', '-es', default=0.1,
              help="Initial epsilon value.")
@click.option('--epsilon_final', '-ef', default=0.0,
              help="Final epsilon value.")
@click.option('--learning_rate', '-lr', default=0.1,
              help="Learning rate of agents.")
@click.option('--q_table_directory', '-d', default=qtable_dir,
              help="Path to directory where q tables will be stored.")
@click.option('--output_directory', '-o', default=output_dir,
              help="Path to directory where output files will be stored.")
@click.option('--logging_directory', '-o', default=logging_dir,
              help="Path to directory where training logs will be stored.")
@click.option('--train_only', '-to', default=0,
              help="Flag to indicate whether only training will occur.")
@click.option('--num_repeated_runs', '-rr', default=0,
              help="Number of complete training runs to execute.")
@click.option('--num_test_trials', '-tt', default=0,
              help="Number of test trials to run for each iteration."
                   "Defaults to number train iterations when not set.")
@click.option('--continue_run', '-cr', default=0,
              help="Run to continue from.")
def train(num_agents, num_opponents, num_iterations, trials_per_iteration, port,
          epsilon_start, epsilon_final, learning_rate, q_table_directory,
          output_directory, logging_directory, train_only, num_repeated_runs,
          num_test_trials, continue_run):
    if not continue_run:
        vars_string = "_agents"+str(num_agents)+"_opponents"+str(num_opponents)+ \
                    "_eps"+str(epsilon_start)+"_lr"+str(learning_rate)+"/"
        q_table_directory += vars_string
        output_directory += vars_string
        logging_directory += vars_string

    for run_index in trange(int(continue_run), int(num_repeated_runs)):
        q_table_dir = os.path.join(q_table_directory, 'run_' + str(run_index))
        os.makedirs(q_table_dir, exist_ok=True)

        output_dir = os.path.join(output_directory, 'run_' + str(run_index))
        os.makedirs(output_dir, exist_ok=True)

        logging_dir = os.path.join(logging_directory, 'run_' + str(run_index))
        os.makedirs(logging_dir, exist_ok=True)

        epsilon_reduce_value = (epsilon_start - epsilon_final) / num_iterations

        if num_test_trials == 0:
            num_test_trials = trials_per_iteration

        for iteration in tqdm(range(0, num_iterations)):
            in_q_table_path = None
            if iteration != 0:
                in_q_table_path = os.path.join(q_table_dir, "iter_" + str(iteration-1)) # type: str
            out_q_table_path = os.path.join(q_table_dir, "iter_" + str(iteration))  # type: str
            os.mkdir(out_q_table_path)

            # initialize q table files
            open(os.path.join(out_q_table_path, 'q_learner1.npy'), 'w+').close()
            open(os.path.join(out_q_table_path, 'q_learner2.npy'), 'w+').close()

            output_file_name = os.path.join(output_dir, 'train_iter_' + str(iteration) + '.txt')

            # set epsilon value
            epsilon_value = epsilon_start - (iteration * epsilon_reduce_value)

            with open(output_file_name, 'w+') as output_file:
                hfo_process = start_hfo_server(
                    num_agents, num_opponents, trials_per_iteration, output_file, port
                )

                log_file_names = []
                open_log_files = []
                for agent_index in range(0, num_agents):
                    logging_file = os.path.join(
                        logging_dir, 'train_iter_' + str(iteration) + 
                        '_player' + str(agent_index + 1) + '.txt')
                    log_file_names.append(logging_file)
                    open_log_files.append(open(logging_file, 'w+'))

                for agent_index in range(0, num_agents):
                    log_file = open_log_files[agent_index]
                    start_player(
                        num_agents-1, num_opponents, agent_index, trials_per_iteration, log_file,
                        epsilon_value, learning_rate, in_q_table_path, out_q_table_path, port=port
                    )
                    time.sleep(10)

                hfo_process.wait()
                close_logs(open_log_files)

            clean_keep_lines(logging_dir, log_file_names, 15)
            clean_keep_lines(output_dir, [output_file_name], 20)

        if not train_only:
            test(
                num_agents, num_opponents, num_iterations, num_test_trials,
                q_table_dir, output_dir, port
            )
        #plot_data(output_directory, vars_string, num_iterations, trials_per_iteration)


def test(num_agents, num_opponents, num_iterations, num_test_trials,
         q_table_directory, output_directory, port):
    for iteration in tqdm(range(0, num_iterations)):
        in_q_table_path = q_table_directory + "/iter_" + str(iteration)

        output_file_name = os.path.join(output_directory,
                                        'test_iter_' + str(iteration) + '.txt')

        with open(output_file_name, "w+") as output_file:
            hfo_process = start_hfo_server(
                num_agents, num_opponents, num_test_trials, output_file, port=port
            )

            for agent_index in range(0, num_agents):
                start_player(
                    num_agents-1, num_opponents, agent_index,
                    num_test_trials, in_q_table_path=in_q_table_path,
                    testing=True, port=port
                )
                time.sleep(10)

            hfo_process.wait()

        clean_keep_lines(output_directory, [output_file_name], 20)


def clean_keep_lines(logging_directory, logs, num_lines):

    temppath = os.path.join(logging_directory, 'tempfile.txt')

    for log_path in logs:
        with open(temppath, 'w+') as tempfile:
            subprocess.run(['tail', '-n', str(num_lines), log_path], stdout=tempfile)
        subprocess.run(['mv', temppath, log_path])


def start_hfo_server(num_agents, num_opponents, num_trials, output_file: typing.IO=subprocess.DEVNULL, port=6000):
    hfo_process = subprocess.Popen(args=['./bin/HFO',
                                         '--port=' + str(port),
                                         '--offense-agents=' + str(num_agents),
                                         '--defense-npcs=' + str(num_opponents),
                                         '--offense-on-ball', '1',
                                         '--trials', str(num_trials),
                                         '--headless', '--no-logging'],
                                   stdout=output_file)
    time.sleep(15)
    return hfo_process


def start_player(num_teammates, num_opponents, agent_index,
                 trials_per_iteration, log_file:typing.IO=subprocess.DEVNULL,
                 epsilon_value=0.0, learning_rate=0.0, in_q_table_path=None,
                 out_q_table_path=None, testing=False, port=6000):
    if testing:
        subprocess.Popen(args=['python', TEST_Q_AGENT_PATH,
                               '--port=' + str(port),
                               '--numTeammates=' + str(num_teammates),
                               '--numOpponents=' + str(num_opponents),
                               '--playerIndex=' + str(agent_index + 1),
                               '--qTableDir=' + in_q_table_path + "/q_learner",
                               '--numEpisodes=' + str(trials_per_iteration)],
                         stdout=log_file)

    else:
        if in_q_table_path:
            subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH,
                                   '--port=' + str(port),
                                   '--numTeammates=' + str(num_teammates),
                                   '--numOpponents=' + str(num_opponents),
                                   '--playerIndex=' + str(agent_index + 1),
                                   '--inQTableDir=' + in_q_table_path + "/q_learner",
                                   '--outQTableDir=' + out_q_table_path + "/q_learner",
                                   '--numEpisodes=' + str(trials_per_iteration),
                                   '--learningRate=' + str(learning_rate),
                                   '--epsilon=' + str(epsilon_value)],
                             stdout=log_file)
        else:
            subprocess.Popen(args=['python', TRAIN_Q_AGENT_PATH,
                                   '--port=' + str(port),
                                   '--numTeammates=' + str(num_teammates),
                                   '--numOpponents=' + str(num_opponents),
                                   '--playerIndex=' + str(agent_index + 1),
                                   '--outQTableDir=' + out_q_table_path + "/q_learner",
                                   '--numEpisodes=' + str(trials_per_iteration),
                                   '--learningRate=' + str(learning_rate),
                                   '--epsilon=' + str(epsilon_value)],
                             stdout=log_file)
    time.sleep(10)


def clean_dir_keep_lines(logging_directory, num_lines):

    files_in_dir = []

    for file in os.listdir(logging_directory):
        files_in_dir.append(os.path.abspath(file))

    clean_keep_lines(logging_directory, files_in_dir, num_lines)


def close_logs(log_file_list):
    for file in log_file_list:
        file.close()

if __name__ == '__main__':
    train()
