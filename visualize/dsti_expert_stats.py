import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from utils import retrieve_final


def get_default_args():
    default_args = OmegaConf.create()
    default_args.runs_dir = (
            Path.cwd() / "runs"
    )  # Root dir where experiment data was saved.
    default_args.exp_names = (
        []
    )  # Unique experiment names to visualize the results for (excluding exp_id).
    default_args.exp_ids = [0]  # Experiment ids.
    default_args.display_names = (
        None  # Pretty display names that will be used when generating the plot.
    )
    default_args.output_dir = Path.cwd() / "figures"  # Target directory.
    return default_args


def main(args):
    logging.basicConfig(
        format=(
            "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] "
            "%(message)s"
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    display_names = args.exp_names if args.display_names is None else args.display_names
    assert len(args.exp_names) == len(display_names)
    output_dicts = []
    for exp_name, display_name in zip(args.exp_names, display_names):
        for exp_id in args.exp_ids:
            run_name = f"{exp_name}_{exp_id}"
            logging.info(f"Processing for: {run_name}")
            final_results = retrieve_final(args, run_name)
            exp_dict = {
                "exp_name": exp_name,
                "exp_id": exp_id,
                "display_name": display_name,
            }
            if "expert_average_costs" in final_results:
                exp_dict["expert_average_costs"] = final_results["expert_average_costs"]
            if "expert_utilization" in final_results:
                exp_dict["expert_utilization"] = final_results["expert_utilization"]
            if "total_experts_used" in final_results:
                exp_dict["total_experts_used"] = final_results["total_experts_used"]
            output_dicts.append(exp_dict)
    save_path = args.output_dir / f"expert_stats.pth"
    torch.save(output_dicts, save_path)
    logging.info(f"Expert stats saved in {str(save_path)}")


if __name__ == "__main__":
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    main(args)
