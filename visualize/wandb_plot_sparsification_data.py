from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from tqdm import tqdm

sns.set_style("darkgrid")

BASELINE_NAME = 'GPT2 OpenAI checkpoint'
ENFORCEMENT_NAME_GELU = 'Sparsity Enforcement, GELU'
ENFORCEMENT_NAME_RELU = 'Sparsity Enforcement, ReLU'
FINETUNING_GELU = 'Finetuning, GELU'
FINETUNING_RELU = 'Finetuning, RELU'

colors = {
    ENFORCEMENT_NAME_GELU: 'tab:blue',
    ENFORCEMENT_NAME_RELU: 'tab:green',
    BASELINE_NAME: 'tab:purple',
    FINETUNING_GELU: 'tab:orange',
    FINETUNING_RELU: 'tab:red',
}
styles = {
    ENFORCEMENT_NAME_GELU: '-',
    ENFORCEMENT_NAME_RELU: '-',
    BASELINE_NAME: '--',
    FINETUNING_GELU: '--',
    FINETUNING_RELU: '--',
}
dsti_width = 1.2
base_width = 0.5
widths = {BASELINE_NAME: base_width,
          ENFORCEMENT_NAME_GELU: dsti_width,
          ENFORCEMENT_NAME_RELU: dsti_width,
          FINETUNING_RELU: base_width,
          FINETUNING_GELU: base_width}


def plot_runs():
    api = wandb.Api()
    runs = list(api.runs("ideas_cv/ztw", filters={"config.dataset": {"$in": ['openwebtext']}}))
    if len(runs) == 0:
        raise ValueError("No runs found for openwebtext")

    total_runs = len(runs)
    runs = [run for run in runs if run.state == 'finished']
    if len(runs) != total_runs:
        print(f"Found {total_runs - len(runs)} unfinished runs for openwebtext")
    outs = [process_run(run) for run in tqdm(runs)]
    outs = [o for o in outs if o['model_class'] in ('gpt2', 'enforce_sparsity') and o['loss'] is not None]
    outs = [o for o in outs if o['lr'] == 6e-4]

    dsti_outs = [o for o in outs if o['model_class'] == 'enforce_sparsity' and o['dsti_weight'] != 0.]
    dsti_enforce_sparsity_vals = list(set([o['dsti_weight'] for o in dsti_outs]))

    gpt2_outs = [o for o in outs if o['model_class'] == 'gpt2']
    gpt2_outs = [
        {
            **gpt2_outs[0], 'dsti_weight': dsti_weight,
        } for dsti_weight in dsti_enforce_sparsity_vals
    ]

    finetuning_outs = [o for o in outs if o['model_class'] == 'enforce_sparsity' and o['dsti_weight'] == 0.]
    finetuning_outs = [
        {
            **out, 'dsti_weight': dsti_weight,
        } for out in finetuning_outs for dsti_weight in dsti_enforce_sparsity_vals
    ]

    df = pd.DataFrame(gpt2_outs + dsti_outs + finetuning_outs)

    fig, axs = plt.subplots(nrows=2, ncols=1)

    loss_plot = sns.lineplot(data=df, x='dsti_weight', y='loss', hue='label', ax=axs[0],
                             # style='label',
                             # style_order=styles,
                             size='label',
                             sizes=widths,
                             palette=colors,
                             )
    loss_plot.set_xlabel(None)
    loss_plot.set(xscale='log')
    loss_plot.set(yscale='log')
    loss_plot.legend_.set_title(None)
    loss_plot.set_ylabel('Loss')

    sparsity_plot = sns.lineplot(data=df, x='dsti_weight', y='sparsity', hue='label', ax=axs[1], legend=False,
                                 # style='label',
                                 # style_order=styles,
                                 size='label',
                                 sizes=widths,
                                 palette=colors,
                                 )
    sparsity_plot.set(xscale='log')
    sparsity_plot.set_xlabel('Sparsity enforcement weight')
    sparsity_plot.set_ylabel('Sparsity')

    output_dir = Path(__file__).parent.parent / 'analysis' / 'gpt2_sparsification'
    output_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig((output_dir / 'gpt2_sparsification.png'))
    fig.savefig((output_dir / 'gpt2_sparsification.pdf'))


def process_run(run):
    config = run.config
    summary = run.summary

    model_class = config['model_class']
    optimizer_args = eval(config['optimizer_args'])
    # model_args = eval(config['model_args'])
    dsti_weight = config.get('dsti_enforce_weight', None)
    dsti_mode = config.get('dsti_enforce_mode', None)
    num_steps = config.get('last_batch', None)
    lr = optimizer_args['lr']

    label = None
    if model_class == 'gpt2':
        label = BASELINE_NAME
    else:
        if dsti_mode is not None:
            if 'gelu' in dsti_mode:
                label = ENFORCEMENT_NAME_GELU
            else:
                label = ENFORCEMENT_NAME_RELU
    if dsti_weight == 0:
        if dsti_mode is not None:
            if 'gelu' in dsti_mode:
                label = FINETUNING_GELU
            else:
                label = FINETUNING_RELU

    output = {
        "model_class": model_class,
        'lr': lr,
        'dsti_weight': dsti_weight,
        'dsti_mode': dsti_mode,
        'num_steps': num_steps,
        'sparsity': summary.get('Eval/Test sparsity', None),
        'loss': summary.get('Eval/Test loss', None),
        'label': label
    }
    return output


def pull_metric(metric_name, run):
    return [log[metric_name] for log in run.scan_history(keys=[metric_name], page_size=100000000)]


if __name__ == "__main__":
    plot_runs()
