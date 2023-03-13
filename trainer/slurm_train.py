import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
from glob import glob

import hydra
import os.path as osp

from GSScheduler.scripts import generate
from GSScheduler.evaluation.FID.fid_score import calculate_fid_given_paths
from hydra.utils import instantiate

from utils import track_jobs_with_pbar
import submitit
from datasets import load_from_disk, concatenate_datasets, Dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logging.Formatter.converter = time.gmtime  # Enforce UTC timestamps
logger = logging.getLogger("fairseq_cli.train")


def load_sharded_hf_ds(data_path):
    logger.info(f"Starting to load data shards from {data_path}")

    ds_list = [load_from_disk(osp.join(data_path, p)) for p in tqdm(paths)]
    logger.info(f"Finished loading dataset: {len(ds_list)} shards with {sum(len(ds) for ds in ds_list)} examples.")
    return ds_list


def generate_shards(text, image_ids, gs_scheduler, executor, cfg):

    output_dirs = []
    jobs = []

    with executor.batch():
        for shard_idx in range(0, math.ceil(len(text) / cfg.data.num_examples_per_shard)):
            start_idx = shard_idx * cfg.data.num_examples_per_shard
            end_idx = start_idx + cfg.data.num_examples_per_shard
            cur_texts = text[start_idx: end_idx]
            cur_image_ids = image_ids[start_idx: end_idx]
            output_dir = os.path.join(cfg.data.dataset_out_dir, str(shard_idx))
            output_dirs.append(output_dir)
            if osp.exists(output_dir):
                logger.info(f"{output_dir} exists. Continuing.")
                continue
            job = executor.submit(
                generate.main,
                texts=cur_texts,
                image_ids=cur_image_ids,
                model_name_or_path=f"{cfg.model.company}/{cfg.model.model_name}",
                out_dir=output_dir,
                images_out_dir=cfg.data.images_out_dir,
                gs_scheduler=gs_scheduler,
                num_inference_steps=cfg.generation.num_inference_steps,
                mixed_precision=cfg.model.mixed_precision,
                use_xformers=cfg.model.use_xformers,
            )
            jobs.append(job)
    track_jobs_with_pbar(jobs)
    return output_dirs


def join_shards(gen_out_dir, output_dirs):
    joined_dir = gen_out_dir + "joined"
    logger.info(f"Joining shards from {gen_out_dir} to {joined_dir}")
    os.makedirs(joined_dir, exist_ok=True)
    for output_dir in tqdm(output_dirs):
        base_out_dir = os.path.basename(output_dir)
        files = os.listdir(output_dir)
        for file in files:
            shutil.copy(os.path.join(output_dir, file), os.path.join(joined_dir, f"{base_out_dir}_{file}"))
    return joined_dir


def calc_fid(executor, reference_dir, joined_dir, batch_size, dims):
    job = executor.submit(calculate_fid_given_paths,
                          paths=[reference_dir, joined_dir],
                          batch_size=batch_size,
                          cuda=True,
                          dims=dims)

    track_jobs_with_pbar([job])

    fid_value = job.result()
    return fid_value


def calc_clip_score(gen_out_dir):
    ds_paths = sorted(glob(f"{gen_out_dir}/*"))
    ds = concatenate_datasets([load_from_disk(ds_path) for ds_path in tqdm(ds_paths)])
    ds.set_format("numpy")
    clip_scores = ds['clip_score']
    final_clip_score = clip_scores.mean()
    return final_clip_score


@hydra.main(config_path="../configs", config_name="eval_config")
def main(cfg: DictConfig):
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=5900, stdoutToServer=True, stderrToServer=True)
    hash_id = hashlib.md5(json.dumps(OmegaConf.to_container(cfg), sort_keys=True).encode('utf-8')).hexdigest()
    cfg.data.results_dir = os.path.join(cfg.data.results_dir, hash_id)
    logger.info(f"Results dir: {cfg.data.results_dir}")
    results_path = f"{cfg.data.results_dir}/config.yaml"
    if os.path.exists(results_path):
        logger.info(f"Results already exist at {results_path}. Exiting.")
        return
    logger.info(f"Final results will be saved to {results_path}")
    slurm_kwargs = {
        "slurm_job_name": cfg.slurm.job_name,
        "slurm_partition": cfg.slurm.partition,
        "slurm_nodes": 1,
        "slurm_additional_parameters": {
            "ntasks": cfg.slurm.n_processes,
            "gpus": cfg.slurm.n_processes,
            "tasks_per_node": cfg.slurm.n_processes,
        },
        "slurm_cpus_per_task": 5,
        "slurm_time": cfg.slurm.time_limit,
        "slurm_exclude": cfg.slurm.exclude if cfg.slurm.exclude else "",
        "stderr_to_stdout": True,
        "slurm_mem": "10GB",
    }

    executor = submitit.AutoExecutor(folder=os.path.join("logs", cfg.slurm.job_name))  # , cluster="debug")
    executor.update_parameters(**slurm_kwargs)

    logger.info("Loading Dataset")
    ds = load_from_disk(cfg.data.dataset_dir)
    if cfg.data.split is not None:
        ds = ds[cfg.data.split]

    logger.info(f"Starting to generate shards from {cfg.data.dataset_dir} to {cfg.data.images_out_dir}")
    texts = ds[cfg.data.dataset_text_col]
    if cfg.data.dataset_image_id_col is not None:
        image_ids = ds[cfg.data.dataset_image_id_col]
    else:
        image_ids = list(range(len(texts)))

    gs_scheduler = instantiate(cfg.generation.gs_scheduler)

    output_dirs = generate_shards(
        texts,
        image_ids,
        gs_scheduler,
        executor,
        cfg,
    )

    if any(not os.path.exists(output_dir) for output_dir in output_dirs):
        raise RuntimeError("Some shards were not generated successfully. Run this script again. Exiting.")

    logger.info("Finished generating shards")

    if cfg.metrics.fid:
        logger.info(f"Calculating FID between {cfg.data.reference_dir} and {cfg.data.images_out_dir}")
        fid_value = calc_fid(
            executor,
            cfg.data.reference_dir,
            cfg.data.images_out_dir,
            cfg.evaluation.batch_size,
            cfg.evaluation.dims
        )
        cfg.results.fid = float(fid_value)
        logger.info(f"Finished calculating FID between {cfg.data.reference_dir} and {cfg.data.images_out_dir} - FID = {fid_value}")
    else:
        logger.info("Skipping FID calculation")

    if cfg.metrics.clip_score:
        logger.info(f"Calculating Clip-Score between text and generated images")
        clip_score = calc_clip_score(cfg.data.dataset_out_dir)
        logger.info(f"Finished calculating Clip-Score between text and generated images = {clip_score}")
        cfg.results.clip_score = float(clip_score)
    else:
        logger.info("Skipping Clip-Score calculation")

    logger.info(f"Writing result to {cfg.data.results_dir}")

    os.makedirs(osp.dirname(cfg.data.results_dir), exist_ok=True)

    OmegaConf.save(cfg, results_path)
    logger.info("Finished pipeline - Trilili Tralala")
    time.sleep(60)


if __name__ == '__main__':
    main()
