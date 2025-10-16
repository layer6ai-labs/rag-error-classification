# rag-error-classification
Framework for RAG and RAGEC.

# Getting Started
1. Create virtual environment using `venv`.
```sh
python3.10 -m venv ./venv310
```
2. Activate the virtual environment.
```sh
source ./venv310/bin/activate
```
3. Install dependencies
```sh
pip install -e .
```
4. Setup OpenAI key. Look at [.envrc.example](.envrc.example) and rename or add an `.envrc` file.
5. Download and preprocess Dragonball and CLAPnq. Note that the shell script contains python calls.
Edit the shell script if necessary.
```sh
bash ./download_data.sh
```
6. Run Dragonball.
```sh
python -m scripts.dragonball_run
```

7. Run CLAPnq
```sh
python -m scripts.clapnq_run
```

## Artifact Managements
The config is specified in `./conf/`. We use `hydra` to parse the config. See [here](baseconfig.py) for a description of the config.

## Components and artifacts

As you can see from the [code](clients/local_rag_client.py), it is decomposed into **components**. Each component will have artifacts as inputs and outputs.
**Artifacts** are data or results that could be saved to or loaded from the disk. An artifact contains the underlying data and the path of where it should be
saved or loaded.

One can "dry run" a component, i.e. to prepare the output artifact(s) of that component, without running the component itself. This allows easy continuation of
the running a sequence of components.


## Logging and artifacts

For each run, the log of the run is saved at `./outputs/{RUN_DATE}/{RUN_TIME}/`. In the directory, you can also see the `.hydra` folder directory that
indicates the config for that run as well. The **artifact** will be shared across runs. The path of the artifact will be specified as `artifact_path` in the main config.
The default path is `./outputs/{DATA}/{RUN_NAME}/`. It will be changed to include the dataset names later.
