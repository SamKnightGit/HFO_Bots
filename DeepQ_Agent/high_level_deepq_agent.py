from hfo import *
import argparse
import os
import gi
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk

def get_screenshot(filename):
    """
    To get correct location use:
    this_dir = os.path.dirname(os.path.realpath(__file__))
    get_screenshot(this_dir + '/screenshots/capture' + str(episode) + ".png")

    :param filename:
    :return:
    """
    screen = Gdk.get_default_root_window().get_screen()
    window = screen.get_active_window()
    pb = Gdk.pixbuf_get_from_window(window, *window.get_geometry())
    pb.savev(filename, "png", (), ())




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--numTeammates', type=int, default=1)
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numEpisodes', type=int, default=1)
    args = parser.parse_args()