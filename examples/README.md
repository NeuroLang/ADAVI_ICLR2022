# ADAVI examples

## Directory organization
* scripts in this directory are related to various experiments performed in our submitted paper
* subdirectory `generative_hbms` contains tensorflow-probability distributions used as input for our ADAVI architecture. This distinction is to clarify the input _descriptors_ necessary for the automatic derivation of our architecture
* subdirectory `juho_lee_set_transformer` contains for convenience a copy of PyTorch scripts from the original authors of SetTransformer, useful for SBI scripts to run (see `README` inside the subdirectory) (authors didn't provide another way to distribute their code)
* subdirectory `baselines` contains the observed data encoder `SummaryNetwork2Plates` described in **sup. mat. F**

## Experiments

All scripts naming follow the same pattern:
* the HBM is indicated at first (`GRE_*` refers to the model from **sec 3.3**, `GM_*` refers to the model from **sec 3.4**, `NC_*` refers to the model from **sec 3.2**, `MSHBM_*` to the toy experiment presented in **sup.mat. E.3**)
* then the method is indicated (see **sup. mat. F** for more details)

As an exmaple, `GRE_ADAVI.py` refers to the application of our method ADAVI to the Gaussian random effects example.

All scripts fit the given method on the given HBM's data, then produce a loss, some samples, and store away results in `../data/`. In addition, scripts display the inference results on a few data points.

Scripts can be ran in interactive mode (as they are `# %%`-decorated), or as `python` scripts **FROM THIS DIRECTORY** (not to break relative paths), in which case:
* for *non-amortized* methods the data index of the point the method is fit upon can be fed using the `--na-val-idx` command line argument
* for the `GRE_*` scripts, the number of groups $G$ can be fed using the `--G` command line argument. Baware that $G=300$ results in significant ressource usage

For instance:
```bash
python GRE_CF-NA.py --G 30 --na-val-idx 4
```
launches a non-amortized Cascading Flows on the $4^{th}$ Gaussian random effects' data point with $G=30$ groups.

## Note on HCP data usage
We cannot freely distribute data from the [Human Connectome Project](https://www.humanconnectome.org/). To run our MS-HBM at scale, and to visualize our results, one would typically need to have access to the HCP dataset, which in our understanding goes beyond the scope of this review process.

As a consequence, experiments related to the application of our architecture to HCP data are not presented here. Nonetheless, the script `MSHBM_ADAVI.py` constitutes a good proxy for it, and leads to interpretable graphical results.