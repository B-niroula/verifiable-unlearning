import argparse
import json
import os
import shutil
from pathlib import Path
from jinja2 import Template
from circ import CirC

DEFAULT_CONFIG = {
    'circ_path': Path('/root/circ'),
    'circuit_dir': Path('/root/verifiable-unlearning/templates/nn_steps'),
    'working_dir': Path('/tmp/nn_steps'),
    'precision': 1000,
    'no_features': 2,
    'no_neurons': 2,
    'lr': 10,
    'debug': False,
}

WEIGHTS = {
    'w0': [[1, 2], [3, 4]],
    'b0': [0, 0],
    'w1': [5, 6],
    'b1': 0,
}

SIG_PARAMS = {
    'W0': int(0.5 * DEFAULT_CONFIG['precision']),
    'W1S': int(0.1501 * DEFAULT_CONFIG['precision']),
    'W3': int(0.0016 * DEFAULT_CONFIG['precision']),
}


def render_step(step: str, cfg):
    tmpl_path = cfg['circuit_dir'].joinpath(f"{step}.template")
    template = Template(tmpl_path.read_text())
    return template.render(
        precision=cfg['precision'],
        no_features=cfg['no_features'],
        no_neurons=cfg['no_neurons'],
        lr=cfg['lr'],
        w0=WEIGHTS['w0'],
        w1=WEIGHTS['w1'],
        b0=WEIGHTS['b0'],
        b1=WEIGHTS['b1'],
        **SIG_PARAMS,
    )


def params_for_step(step: str, cfg):
    if step == 'forward':
        return [('private', 'x', f'u64[{cfg["no_features"]}]', [1 for _ in range(cfg['no_features'])])]
    if step == 'backward':
        return [
            ('private', 'x', f'u64[{cfg["no_features"]}]', [1 for _ in range(cfg['no_features'])]),
            ('private', 'y', 'u64', cfg['precision']),
        ]
    if step == 'update':
        return [
            ('private', 'dw0', f'u64[{cfg["no_neurons"]}][{cfg["no_features"]}]', [[1]*cfg['no_features'] for _ in range(cfg['no_neurons'])]),
            ('private', 'dw1', f'u64[{cfg["no_neurons"]}]', [1]*cfg['no_neurons']),
            ('private', 'db0', f'u64[{cfg["no_neurons"]}]', [1]*cfg['no_neurons']),
            ('private', 'db1', 'u64', 1),
        ]
    raise ValueError(step)


def compile_step(step: str, cfg):
    working_dir = cfg['working_dir'].joinpath(step)
    working_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent
    poseidon_src = repo_root.joinpath('templates', 'poseidon')
    if poseidon_src.exists():
        shutil.copytree(poseidon_src, working_dir.joinpath('poseidon'), dirs_exist_ok=True)
    stdlib_src = Path(os.environ.get('CIRC_STDLIB', '/root/circ/stdlib'))
    if stdlib_src.exists():
        shutil.copytree(stdlib_src, working_dir.joinpath('stdlib'), dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"std library not found: {stdlib_src}")
    circ = CirC(cfg['circ_path'], debug=cfg['debug'])
    proof_src = render_step(step, cfg)
    working_dir.joinpath('circuit.zok').write_text(proof_src)
    params = params_for_step(step, cfg)
    circ.spartan_nizk(params, working_dir)
    log = working_dir.joinpath('circ.log.txt').read_text()
    for line in log.splitlines():
        if 'final R1CS size:' in line:
            print(f"{step} constraints: {line.split(':')[-1].strip()}")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['forward', 'backward', 'update'], required=True)
    args = parser.parse_args()
    cfg = DEFAULT_CONFIG.copy()
    compile_step(args.step, cfg)


if __name__ == '__main__':
    main()
