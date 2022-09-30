import os
import setuptools


with open("requirements.txt", mode="r", encoding="utf-8") as f_in:
    install_requires = [line.strip() for line in f_in]


with open(os.path.join("knn_models", "version.txt"), mode="r", encoding="utf-8") as f_in:
    version = f_in.read()


if __name__ == "__main__":
    setuptools.setup(
        name="knn_models",
        version=version,
        description=["k-Nearest Neighbor Augmented Sequence-to-Sequence Toolkit"],
        url="https://github.com/cordercorder/knn-models",
        install_requires=install_requires,
        packages=[
            "knn_models",
            "knn_models.tasks",
            "knn_models.data",
            "knn_models.dataclass",
        ],
        entry_points={
            "console_scripts": [
                "build-faiss-index = knn_models_cli.build_faiss_index:cli_main",
                "count-tokens = knn_models_cli.count_tokens:cli_main",
                "dedup-datastore = knn_models_cli.dedup_datastore:cli_main",
                "eval_knn_lm = knn_models_cli.eval_knn_lm:cli_main",
                "generate_lm_datastore = knn_models_cli.generate_lm_datastore:cli_main",
                "generate_mt_datastore = knn_models_cli.generate_mt_datastore:cli_main",
                "merge_datastore = knn_models_cli.merge_datastore:cli_main",
                "prune_datastore = knn_models_cli.prune_datastore:cli_main",
                "reduce_datastore_dims = knn_models_cli.reduce_datastore_dims:cli_main",
                "tune_knn_params = knn_models_cli.tune_knn_params:cli_main",
            ]
        }
    )
