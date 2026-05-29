# Sex-related structural alterations across common epilepsies: a worldwide ENIGMA study
This repository includes the scripts for sex difference analyses in cortical thickness and subcortical volume data in the dataset in combination of ENIGMA-epilepsy, ECP dataset, MNI dataset. 

- Correspondence to Huantao Wen (wenh@cbs.mpg.de) and Sofie Valk (valk@cbs.mpg.de)
- Max Planck Institute for Human Cognitive and Brain Sciences, Leipzig, Germany
- Institute of Neuroscience and Medicine (INM-7: Brain and Behaviour), Research Center Jülich, Jülich, German

### Scripts
**Data processing**

`data_combat.py` 
- Harmonizes cortical thickness and subcortical volume data across sites using ComBat in common epilepsies (TLE and GGE)

**Main analyses**

`group_diff.ipynb`
- Sex-stratified case-control differences analyses in TLE and GGE
- Sex differences analyses in epilepsy and corresponding healthy controls

`clinic_var_diff.ipynb`
- Sex differences in the effect of disease duration in TLE and GGE
- Sex differences in the effect of age of onset in TLE and GGE

**Supplementary analyses**

`robustness_side_lesion.ipynb`
- Subcohort analyses of sex differences across left and right TLE subgroups
- Sex differences in mesial temporal sclerosis (MTS) and non-lesional TLE patients

`robustness_age.ipynb`
- Sex differences in the effects of age in TLE and GGE

`robustness_across_datasets.ipynb`
- Consistency of sex interaction effects analyses in TLE samples across datasets

**Function**

`data_combat.py` 
- Includes helper functions for data harmonization, main statistical analysis

## 📦 Requirements
- neuroCombat
- ENIMGA toolbox
- BrainStat
- Statsmodel
- SciPy
