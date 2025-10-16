import logging
import pathlib
import traceback

import hydra
from omegaconf import DictConfig, OmegaConf

from baseconfig import LocalRagConfig
from clients.local_rag_client import LocalRagClient


@hydra.main(version_base=None, config_path="../conf/", config_name="config-dragonball")
def main(config: DictConfig):
    for logger in config.loggers_disabled:
        logging.getLogger(logger).disabled = True

    rag_config: LocalRagConfig = OmegaConf.to_object(config.rag)
    data_name = rag_config.data.corpus_name
    out_path = pathlib.Path("outputs") / data_name / config.artifact_run_name
    try:
        rag_client = LocalRagClient(rag_config, out_path)
        rag_client.run(chunk_sample=None, query_sample=None)
        rag_client.evaluate()
        rag_client.classify_errors()
    except Exception as e:
        tb = traceback.format_exception(e)
        for entry in tb:
            lines = entry.split("\n")
            for line in lines:
                if len(line) > 0:
                    logging.error(line)
        raise e


if __name__ == "__main__":
    main()
