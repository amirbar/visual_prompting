import argparse
from multiprocessing import Pool

import pandas as pd
import glob
import logging
import shutil
import os

from PIL import Image
from arxiv import Search, Client
from pdf2image import convert_from_path
import tarfile
import PIL
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)
CHUNK_SIZE = 100
MAX_SIZE = 1024
big_slow_client = Client(
    page_size=CHUNK_SIZE,
    delay_seconds=10,
    num_retries=100
)


def download_paper_from_ids(paper, paper_id, directory, filter_files=None):
    file_location = paper.download_source()
    assert file_location.endswith(".tar.gz")
    return extract_file(file_location, directory, paper_id, filter_files)


def _resize_and_save(path, file, paper_id):
    saved_fp = os.path.join(path, file.name)
    try:
        if file.name.endswith(".pdf"):
            images = convert_from_path(saved_fp)
            output_path = os.path.join(path, f"{file.name}_{paper_id}_0.png")
        else:
            images = [Image.open(saved_fp)]
            output_path = saved_fp

        os.remove(saved_fp)

        if len(images) != 1:
            return

        img = images[0]
        img = img.convert('RGB')
        img.thumbnail((MAX_SIZE, MAX_SIZE))
        img.save(output_path, "PNG")
    except PIL.Image.DecompressionBombError:
        print("DecompressionBombError")


def extract_file(file_location, directory, paper_id, filter_files):
    tar_file = tarfile.open(file_location, 'r:gz')
    path = os.path.join(directory, paper_id)
    os.makedirs(path, exist_ok=True)

    while True:
        next_file = tar_file.next()
        if next_file is None:
            break
        if not _check_file_name(next_file.name):
            continue

        next_file.name = next_file.name.replace('/', '_')
        if filter_files is not None and next_file.name not in filter_files:
            continue
        tar_file.extract(next_file, path)
        # print('Extracted {}'.format(next_file.name))
        _resize_and_save(path, next_file, paper_id)
    tar_file.close()
    os.remove(file_location)
    return path


def url_to_id(url: str) -> str:
    """
    Parse the given URL of the form `https://arxiv.org/abs/1907.13625` to the id `1907.13625`.

    Args:
        url: Input arxiv URL.

    Returns:
        str: ArXiv article ID.
    """
    if url.endswith(".pdf"):
        url = url[:-4]

    return url.split("/")[-1]


def clean_up_path(path):
    for f in glob.glob(os.path.join(path, '*')):
        if os.path.isdir(f):
            try:
                shutil.rmtree(f)
            except OSError as e:
                print(e)


def _check_file_name(file_name: str):
    file_name = file_name.lower()
    file_suffix = file_name.split(".")[-1]
    if file_suffix in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'pdf']:
        if 'arch' in file_name or 'pipeline' in file_name:
            # Remove architecture
            return False
        if 'CVPR_Template' in file_name:
            # Remove templates
            return False
        return True
    return False


def main(args):
    assert args.split in ['train', 'val']
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv('df_train.csv', dtype={'paper_id': str, 'img_name': str})
    agg_df = df.groupby(['paper_id'])['img_name'].apply(list)
    if args.workers > 1:
        with Pool(args.workers) as p:
            print(p.map(download_figures, np.array_split(agg_df, args.workers)))
    else:
        download_figures(agg_df)


def download_figures(agg_df):
    output_dir = os.path.join(args.output_dir, args.split)
    paper_ids = list(agg_df.index)
    num_chunks = len(paper_ids) // CHUNK_SIZE + 1
    for i in tqdm(range(num_chunks)):
        success_counter = 0
        skip_counter = 0
        fail_counter = 0
        for paper in big_slow_client.results(Search(id_list=paper_ids[CHUNK_SIZE * i:CHUNK_SIZE * (i + 1)])):
            paper_id = paper.get_short_id().split('v')[0]
            try:
                if args.skip_exists and os.path.exists(os.path.join(output_dir, paper_id)):
                    skip_counter += 1
                    continue
                download_paper_from_ids(paper, paper_id, output_dir, filter_files=agg_df[paper_id])
                success_counter += 1
            except Exception as e:
                fail_counter += 1
                print(e)
        print(
            f"Downloaded successfully {success_counter}/{CHUNK_SIZE} papers. Skipped {skip_counter}/{CHUNK_SIZE}. Failed to download {fail_counter}/{CHUNK_SIZE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Dataset downloader', add_help=False)
    parser.add_argument('--output_dir', default='dataset', type=str)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--skip_exists', default=True, type=str)
    parser.add_argument('--workers', default=1, type=int)
    args = parser.parse_args()

    main(args)
