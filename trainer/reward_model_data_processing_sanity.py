import logging
import os
import random
from glob import glob

import hydra
import pandas as pd
import psycopg2
from datasets import DatasetDict, Dataset
from img2dataset import download
from omegaconf import DictConfig
from tqdm import tqdm

DATABASE_URL = os.environ["DATABASE_URL"]
BASE_IMAGE_URL = "https://text-to-image-human-preferences.s3.us-east-2.amazonaws.com/images"
logger = logging.getLogger(__name__)

SHORT_SD2_1 = "SD2.1"
SHORT_DP2_0 = "DP2.0"
SHORT_SD2_1_API = "SD2.1-API"

CLS_NAME2SHORT_NAME = {
    "yuvalkirstain/dreamlike-photoreal-2-flax": SHORT_DP2_0,
    "stabilityai/stable-diffusion-2-1": SHORT_SD2_1,
    "stable-diffusion-768-v2-1": SHORT_SD2_1_API,
}

ADDITIONAL_COLUMNS = [
    'caption',
    "ranking_id",
    'created_at',
    'user_id',
    'best_image_uid',
    'bad_image_uid',
    'best_model',
    'bad_model',
    'are_different'
    'has_label'
]

BLOCKED = {
    280, 331, 437, 641, 718, 729, 783, 984, 1023, 1040, 1059, 1149, 1187, 1177, 1202, 1203, 1220,
    1230, 1227, 1279, 1405, 1460, 1623, 1627, 1758, 1801, 1907, 1917, 1922, 2071, 2215, 2239, 2286, 2322, 2357, 2452,
    2459, 2481, 2513, 2515, 2520, 2545, 2596, 2603, 2617, 2638, 2709, 2783, 2842, 2266, 2899
}

NSFW_WORDS = {
    "seductive", "sexy", "nude", "naked", "nudity", "nakedness", "nakedly", "nakedness",
    "busty", "boobs", "gay", "shirtless", "nude", "revealing", "naked", "breasts", "amouranth",
    "nigger", "sussy", "tits", "lingerie", "sex", "bikini", "putin", "nazi", "underwear", "stomach", "xi jinping",
    "thong", "fuck", "lola myluv", "elsa jean", "porn", "courtesan", "b00bs", "undressed", "anal", "blowjo", "p0rn",
    "sxy", "blowjo", "cumshot", "vagina", "horny", "anal", " ass", "pussy", "onlyfans", "crotch",
    "gagg", "nudist", "lecherous", "voluptuous", "buxom", "lustful", "genitals", "nymph", "camgirl", "exposed",
    "babe", "hitler", "nakey", "butt", "suck", "blowjob", "unclothed", "wearing nothing", "breasted",
    "nsfw", "erotic", "hentai", "topless", "nipple", "without clothes", "playboy", "bøøbs", "booty", "leotard",
    "nudist", "scantily clad", "salacious", "minimal clothes", "see through", "bulge", "onahole",
    "loincloth", "man posing", "jockstrap", " hot ", " bra ", "shower", "femdom", "harvey weinstein",
    "emma watson", "curvaceous", "sensuous", "chest", "no clothes", "scantily", "undies", "bulging", "swimsuit",
    "belly", "sweatpants", "suntanning", "naughty", "jada stevens", "strip", "cleavage"
}


def add_bad_image(x):
    if x['best_image_uid'] == "none":
        return "none"
    elif x['image_1_uid'] == x['best_image_uid']:
        return x['image_2_uid']
    return x['image_1_uid']


def fix_model_names(images_df):
    total = 0
    num_fixed = 0
    images_df = images_df.sort_values(by="created_at")
    fixed_images_df = images_df.copy(deep=True)
    first_valid_timestep = fixed_images_df[fixed_images_df.model_id == SHORT_SD2_1_API].iloc[0].created_at
    for i in tqdm(range(0, len(images_df) - 2)):
        total += 1
        image_1 = images_df.iloc[i]
        if image_1.created_at >= first_valid_timestep:
            break
        image_2 = images_df.iloc[i + 1]
        image_3 = images_df.iloc[i + 2]
        if any(image.model_id != image_1.model_id for image in [image_1, image_2, image_3]):
            continue
        if (image_3.created_at - image_2.created_at).seconds > 0.3 or (
                image_2.created_at - image_1.created_at).seconds > 0.3:
            continue
        if any(image.prompt != image_1.prompt for image in [image_1, image_2, image_3]):
            continue
        if image_3.idx != image_1.idx or image_3.idx == image_2.idx:
            continue
        num_fixed += 1
        if image_1.model_id == SHORT_DP2_0:
            fixed_images_df.at[i + 2, 'model_id'] = SHORT_SD2_1
        else:
            fixed_images_df.at[i + 2, 'model_id'] = SHORT_DP2_0
    logger.info(f"Fixed {num_fixed} out of {total} images")
    return fixed_images_df


def shorten_model_name(model_id):
    if model_id in CLS_NAME2SHORT_NAME:
        return CLS_NAME2SHORT_NAME[model_id]
    raise ValueError(f"Unknown model_id: {model_id}")


def remove_non_existing_images(df, image_uid2model_id):
    logger.info(f"Removing non existing images...")
    df = df[df['image_0_uid'].isin(image_uid2model_id)]
    df = df[df['image_1_uid'].isin(image_uid2model_id)]
    logger.info(f"After removing non existing images: {len(df)}")
    return df


def add_models(df, image_uid2model_id):
    df["model_0"] = df.apply(lambda x: image_uid2model_id[x['image_0_uid']], axis=1)
    df["model_1"] = df.apply(lambda x: image_uid2model_id[x['image_1_uid']], axis=1)
    return df


def update_if_different_models(df):
    logger.info(f"adding if the models are different...")
    df["are_different"] = (df["model_0"] != df["model_1"])
    return df


def clean_nsfw(df):
    for word in NSFW_WORDS:
        df = df[~ df["prompt"].str.lower().str.contains(word)]
    df = df[~ df["user_id"].isin(BLOCKED)]
    logger.info(f"After cleaning nsfw: {len(df)}")
    return df


def add_label_0(x):
    if x["best_image_uid"] == x["image_1_uid"]:
        return 0
    elif x["best_image_uid"] == x["image_0_uid"]:
        return 1
    else:
        return 0.5


def add_label_1(x):
    if x["best_image_uid"] == x["image_0_uid"]:
        return 0
    elif x["best_image_uid"] == x["image_1_uid"]:
        return 1
    else:
        return 0.5


def get_all_rankings() -> pd.DataFrame:
    logger.info("Loading DB...")
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cursor = conn.cursor()
    logger.info("Querying DB...")
    cursor.execute(f"SELECT * FROM rankings WHERE created_at >= '2023-02-18'")
    logger.info("Fetching DB...")
    rankings = cursor.fetchall()
    cursor.close()
    conn.close()
    logger.info("Converting to dataframe...")
    df = pd.DataFrame(rankings,
                      columns=['ranking_id', 'created_at', 'user_id', 'image_0_uid', 'image_1_uid', 'image_2_uid',
                               'image_3_uid', 'best_image_uid', 'prompt'])
    df = df[['ranking_id', 'created_at', 'user_id', 'image_0_uid', 'image_1_uid', 'best_image_uid', 'prompt']]
    return df


def add_urls(df):
    logger.info("Adding URLs...")
    df['image_0_url'] = df.apply(
        lambda x: f"{BASE_IMAGE_URL}/{x['image_0_uid']}.png", axis=1
    )
    df['image_1_url'] = df.apply(
        lambda x: f"{BASE_IMAGE_URL}/{x['image_1_uid']}.png", axis=1
    )
    return df


def get_all_images() -> pd.DataFrame:
    logger.info("Loading DB...")
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cursor = conn.cursor()
    logger.info("Querying DB...")
    cursor.execute(f"SELECT * FROM images WHERE created_at >= '2023-02-18'")
    logger.info("Fetching DB...")
    images = cursor.fetchall()
    cursor.close()
    conn.close()
    logger.info("Converting to dataframe...")
    df = pd.DataFrame(images, columns=['image_id',
                                       'created_at',
                                       'image_uid',
                                       'user_id',
                                       'prompt',
                                       'negative_prompt',
                                       'seed',
                                       'gs',
                                       'steps',
                                       'idx',
                                       'num_generated',
                                       'scheduler_cls',
                                       'model_id'])
    return df


def download_images(cfg, url_col, save_additional_cols, out_path):
    logger.info(f"Downloading {url_col} images to {out_path}...")
    if cfg.dataset_columns.text_col in save_additional_cols:
        save_additional_cols.remove(cfg.dataset_columns.text_col)
    if url_col in save_additional_cols:
        save_additional_cols.remove(url_col)
    download(
        processes_count=cfg.download.process_count,
        thread_count=cfg.download.thread_count,
        url_list=cfg.paths.urls_out_path,
        image_size=cfg.download.image_size,
        output_folder=out_path,
        resize_mode=cfg.download.resize_mode,
        resize_only_if_bigger=True,
        output_format="parquet",
        input_format="parquet",
        url_col=url_col,
        caption_col=cfg.dataset_columns.text_col,
        number_sample_per_shard=cfg.download.num_samples_per_shard,
        distributor="multiprocessing",
        min_image_size=cfg.download.image_size,
        save_additional_columns=save_additional_cols,
        max_shard_retry=cfg.download.max_shard_retry,
        retries=cfg.download.retries,
    )


def pd_read_parquets(parquet_path):
    dfs = []
    for path in sorted(glob(f"{parquet_path}/*.parquet")):
        dfs.append(pd.read_parquet(path))
    df = pd.concat(dfs)
    return df


def merge_images(cfg, additional_columns):

    logger.info("Merging image 0 and image 1...")
    df_0 = pd_read_parquets(cfg.paths.images_good_out_path)
    logger.info(f"Num image 0 images: {len(df_0)}")
    df_0[cfg.dataset_columns.url_good_col] = df_0["url"]
    df_0["jpg_0"] = df_0["jpg"]
    df_0 = df_0[df_0["status"] == "success"]
    columns_0 = [col for col in additional_columns if col in df_0.columns]
    df_0 = df_0[['jpg_0', cfg.dataset_columns.output_text_col] + columns_0]

    df_1 = pd_read_parquets(cfg.paths.images_bad_out_path)
    logger.info(f"Num image 1 images: {len(df_1)}")
    df_1[cfg.dataset_columns.url_bad_col] = df_1["url"]
    df_1["jpg_1"] = df_1["jpg"]
    df_1 = df_1[df_1["status"] == "success"]
    columns_1 = [col for col in additional_columns if col in df_1.columns]
    df_1 = df_1[['jpg_1', cfg.dataset_columns.output_text_col] + columns_1]

    inter_columns = sorted(set(df_0.columns) & set(df_1.columns))
    df = df_0.merge(
        df_1,
        on=inter_columns,
    )
    union_columns = sorted(set(df_0.columns) | set(df_1.columns))
    df = df[union_columns]
    logger.info(f"Num merged rankings: {len(df)}")
    return df


def pd_fix_model_names(x, first_valid_timestep, images_df):
    if x.created_at >= first_valid_timestep:
        return x.model_id
    if x.name - 2 < 0:
        return x.model_id

    image_3 = images_df.iloc[x.name]
    image_2 = images_df.iloc[x.name - 1]
    image_1 = images_df.iloc[x.name - 2]

    if any(image.model_id != image_1.model_id for image in [image_1, image_2, image_3]):
        return x.model_id
    if (image_3.created_at - image_2.created_at).seconds > 0.3 or (
            image_2.created_at - image_1.created_at).seconds > 0.3:
        return x.model_id
    if any(image.prompt != image_1.prompt for image in [image_1, image_2, image_3]):
        return x.model_id
    if image_3.idx != image_1.idx or image_3.idx == image_2.idx:
        return x.model_id

    if image_1.model_id == SHORT_DP2_0:
        return SHORT_SD2_1
    else:
        return SHORT_DP2_0


@hydra.main(config_path="conf", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Starting...")
    if not os.path.exists(cfg.paths.urls_out_path):
        df = get_all_rankings()
        logger.info(f"Initial number of rankings: {len(df)}")
        df["label_0"] = df.apply(lambda x: add_label_0(x), axis=1)
        df["label_1"] = df.apply(lambda x: add_label_1(x), axis=1)
        df["has_label"] = (~ df["best_image_uid"].str.contains("none"))

        df = clean_nsfw(df)

        df["prompt"] = df.apply(lambda x: x["prompt"].replace("[single_quote]", '\''), axis=1)

        df = add_urls(df)

        images_df = get_all_images()
        images_df = images_df.sort_values(by="created_at")
        images_df = images_df.reset_index()
        images_df["model_id"] = images_df.apply(lambda x: shorten_model_name(x.model_id), axis=1)
        first_valid_timestep = images_df[images_df.model_id == SHORT_SD2_1_API].iloc[0].created_at

        fixed_images_df = images_df.copy(deep=True)
        logger.info("Fixing model names...")
        fixed_images_df["model_id"] = fixed_images_df.apply(lambda x: pd_fix_model_names(x, first_valid_timestep, images_df), axis=1)
        images_df = fixed_images_df
        logger.info("Getting image_uid to model_id mapping...")
        image_uid2model_id = dict(zip(fixed_images_df.image_uid, images_df.model_id))
        df = remove_non_existing_images(df, image_uid2model_id)
        df = add_models(df, image_uid2model_id)
        df = update_if_different_models(df)


        logger.info(f"Saving urls for img2dataset - {len(df)} rows...")
        os.makedirs(os.path.dirname(cfg.paths.urls_out_path), exist_ok=True)
        df.to_parquet(cfg.paths.urls_out_path)
    else:
        logger.info(f"Skipping download of urls, already exists: {cfg.paths.urls_out_path}")
        df = pd.read_parquet(cfg.paths.urls_out_path)

    if not os.path.exists(cfg.paths.images_good_out_path):
        download_images(
            cfg,
            "image_0_url",
            df.columns.to_list(),
            cfg.paths.images_good_out_path
        )
    else:
        logger.info(f"Skipping download of image 0, already exists: {cfg.paths.images_good_out_path}")

    if not os.path.exists(cfg.paths.images_bad_out_path):
        download_images(
            cfg,
            "image_1_url",
            df.columns.to_list(),
            cfg.paths.images_bad_out_path
        )
    else:
        logger.info(f"Skipping download of image 1, already exists: {cfg.paths.images_bad_out_path}")

    df = merge_images(cfg, df.columns.to_list())
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info("Splitting dataset by prompt...")
    non_tie_prompts = list(df[df["has_label"]][cfg.dataset_columns.output_text_col].unique())
    random.shuffle(non_tie_prompts)
    test_prompts = set(non_tie_prompts[:cfg.split.test_size])
    train_df = df[~df[cfg.dataset_columns.output_text_col].isin(test_prompts)]
    test_df = df[df[cfg.dataset_columns.output_text_col].isin(test_prompts)]
    test_df = test_df[test_df["has_label"]]
    test_prompts = set(list(test_prompts)[:len(test_prompts) // 2])
    val_df = test_df[~test_df[cfg.dataset_columns.output_text_col].isin(test_prompts)]
    test_df = test_df[test_df[cfg.dataset_columns.output_text_col].isin(test_prompts)]
    unique_val_df = val_df.drop_duplicates(subset=[cfg.dataset_columns.output_text_col])
    unique_test_df = test_df.drop_duplicates(subset=[cfg.dataset_columns.output_text_col])
    user_unique_val_df = unique_val_df.drop_duplicates(subset=["user_id"])
    user_unique_test_df = unique_test_df.drop_duplicates(subset=["user_id"])

    logger.info("Creating dataset from dfs...")
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
            "validation_unique": Dataset.from_pandas(unique_val_df),
            "test_unique": Dataset.from_pandas(unique_test_df),
            "validation_unique_user": Dataset.from_pandas(user_unique_val_df),
            "test_unique_user": Dataset.from_pandas(user_unique_test_df),
        }
    )

    logger.info(f"{dataset=}")
    logger.info(f"saving to disk {cfg.paths.dataset_out_path}")
    dataset.save_to_disk(cfg.paths.dataset_out_path, num_proc=cfg.download.process_count)

    # print(f"pushing to hub yuvalkirstain/{cfg.paths.dataset_hub_name}")
    # dataset.push_to_hub(cfg.paths.dataset_hub_name)


if __name__ == '__main__':
    import pydevd_pycharm

    pydevd_pycharm.settrace('localhost', port=5900, stdoutToServer=True, stderrToServer=True)

    main()
