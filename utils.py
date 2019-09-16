import os
import pandas as pd
import pickle

from argparse import ArgumentParser
from shutil import copyfile
from subprocess import check_output

def save_file(obj, fpath):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)

def load_file(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)

def write(fpath, text):
    with open(fpath, 'a+') as f:
        f.write(text + '\n')

def run_cmd(config):
    os.makedirs('scripts', exist_ok=True)
    fpaths = os.listdir('scripts')
    num_scripts = len([fpath for fpath in fpaths if '.sh' in fpath])
    script_fpath = f'robust_{num_scripts}.sh'
    copyfile('template.sh', 'scripts/' + script_fpath)
    with open('scripts/' + script_fpath, 'a') as f:
        f.write('\npython {}'.format(config['cmd']))
    os.chdir('scripts')
    print(check_output(['bash', '-c', f'sbatch --partition Standard {script_fpath}']))

def summary():
    df = {}
    fpaths = os.listdir('results')
    for fpath in fpaths:
        with open(os.path.join('results', fpath, 'results.txt')) as f:
            val, test = f.readlines()
            val, test = float(val.split(' ')[1]), float(test.split(' ')[1])
            df[fpath] = {'val': val, 'test': test}
    df = pd.DataFrame(df).T
    df.sort_values('val')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cmd', type=str, default=None)
    config = parser.parse_args()
    config = vars(config)
    if '--cmd' in config:
        run_cmd(config)
    else:
        summary()