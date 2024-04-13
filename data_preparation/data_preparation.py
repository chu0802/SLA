import zipfile
from ast import literal_eval
from itertools import product
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

import configargparse
from tqdm import tqdm


def download(url, path, auto_mkdir=True):
    if auto_mkdir:
        path.parent.mkdir(exist_ok=True, parents=True)
    response = urlopen(url)
    with tqdm.wrapattr(
        open(path, "wb"), "write", miniters=1, total=response.length
    ) as f:
        for chunk in response:
            f.write(chunk)


def download_OfficeHome(args):
    file_path = args.dataset_dir / "OfficeHome.zip"

    download(args.url, file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(args.dataset_dir)

    (args.dataset_dir / "OfficeHome" / "RealWorld").rename(
        args.dataset_dir / "OfficeHome" / "Real"
    )


def download_DomainNet(args):
    for domain in args.domains:
        file_path = args.dataset_dir / f"{domain}.zip"

        # reference: http://ai.bu.edu/M3SDA/
        if domain in ["clipart", "painting"]:
            url = f"{args.url}/groundtruth/{domain}.zip"
        else:
            url = f"{args.url}/{domain}.zip"

        download(url, file_path)

        dataset_sub_dir = args.dataset_dir / "DomainNet"
        dataset_sub_dir.mkdir(exist_ok=True, parents=True)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dataset_sub_dir)


def prepare_text_list(args):
    # Download source / validation text list
    for (types, domain_type, suffix, output_name), domain in product(
        [
            ("labeled", "source", "", "all.txt"),
            ("validation", "target", "_3", "val.txt"),
        ],
        args.domains,
    ):
        download(
            f"{args.text_url}/{types}_{domain_type}_images_{domain}{suffix}.txt",
            args.dataset_dir / args.dataset / "text" / domain / output_name,
        )

    # Download labeled / unlabeled target text list
    for (types, output_type), domain, num_labels in product(
        [("labeled", "train"), ("unlabeled", "test")],
        args.domains,
        [1, 3],
    ):
        download(
            f"{args.text_url}/{types}_target_images_{domain}_{num_labels}.txt",
            args.dataset_dir
            / args.dataset
            / "text"
            / domain
            / f"{output_type}_{num_labels}.txt",
        )


def arguments_parsing():
    p = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    p.add("--config", is_config_file=True, default="dataset.yaml")
    p.add(
        "--dataset",
        type=str,
        default="OfficeHome",
        choices=["DomainNet", "Office31", "OfficeHome"],
    )
    p.add("--text_url", type=str)
    p.add("--dataset_dir", type=Path)
    p.add("--dataset_cfg", type=literal_eval)

    args = p.parse_args()
    args.url = args.dataset_cfg[args.dataset]["url"]
    args.domains = args.dataset_cfg[args.dataset]["domains"]
    args.text_dir = args.dataset_dir / args.dataset / "text"
    args.text_url = args.text_url + (
        {
            "DomainNet": "multi",
            "Office31": "office",
            "OfficeHome": "office_home",
        }[args.dataset]
    )
    return args


if __name__ == "__main__":
    args = arguments_parsing()
    download_fn = globals()[f"download_{args.dataset}"]
    download_fn(args)
    prepare_text_list(args)
