import click
import tqdm
import subprocess
from .deep_qnetwork import Deep_QNetwork
DEEP_QNETWORK_LOAD_PATH = ''

@click.command()
@click.option('--port', '-p', default=6000)
@click.option('--deep_network_path', '-dn', default=DEEP_QNETWORK_LOAD_PATH)
@click.option('--num_teammates', '-tm', default=2)
@click.option('--num_opponents', '-op', default=2)
@click.option('--trials_per_iteration', '-t', default=1)
@click.option('--num_iterations', '-n', default=0)
def run_testing(port, deep_network_path, num_teammates, num_opponents,
                trials_per_iteration, num_iterations):

    # 0 state dimensions - network should be loaded, not created, for testing.
    deep_network = Deep_QNetwork(0, load_location=DEEP_QNETWORK_LOAD_PATH,
                                 num_teammates=num_teammates,
                                 num_opponents=num_opponents)
    for iteration in num_iterations:
        hfo_process = subprocess.Popen(args=['./bin/HFO', '--offense-agents=2', '--defense-npcs=2',
                                                 '--offense-on-ball', '1', '--trials', trials_per_iteration,
                                                 '--headless'])
        hfo_process.wait()

