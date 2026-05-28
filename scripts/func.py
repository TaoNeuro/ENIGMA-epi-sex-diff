import math
import numpy as np
import pandas as pd
import scipy.stats as ss
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect
from neuroCombat import neuroCombat
from statsmodels.stats.multitest import fdrcorrection

## data harmonization
class data_combat:
    def __init__(self, demo, group_type='allepi', sites_tle=None, sites_ige=None):
        # specify the epilepsy type, and filter the data
        if group_type == 'allepi':
            self.demo = demo
        elif group_type == 'tle':
            if sites_tle is None:
                raise ValueError("sites_tle must be provided when group_type='tle'")
            self.demo = demo[
                demo['Site'].isin(sites_tle) &
                demo['SDx'].isin([0, 3, 4, 5, 6])
                ]
        elif group_type == 'ige':
            if sites_ige is None:
                raise ValueError("sites_ige must be provided when group_type='ige'")
            self.demo = demo[
                demo['Site'].isin(sites_ige) &
                demo['SDx'].isin([0, 2])
            ]
        else:
            raise ValueError(f"Unknown group_type: {group_type}")

        # get the index
        self.index = self.demo.index

    def prepare_covariates(self, site, age, sex, group):
        '''define the variable you want to combat, and the biological covariates for preserve'''                    

        site = self.demo[site].loc[self.index].copy()
        age = self.demo[age].loc[self.index].copy()
        
        sex = self.demo[sex].loc[self.index].copy().to_numpy().astype(float)
        sex[sex == 1] = 1
        sex[sex == 2] = 2
        sex = pd.Series(sex, index=self.index)

        group = self.demo[group].loc[self.index].copy().to_numpy().astype(float)
        group[group == 1] = 1
        group[group == 0] = 2
        group = pd.Series(group, index = self.index)

        return pd.DataFrame({'site':site, 'age':age, 'sex':sex, 'group':group})

    def harmonize_data(self, data_metric, data_cov):
        """Harmonize data using neuroCombat with specified covariates"""
        data = data_metric.loc[self.index]
        covariates = data_cov

        data_combat = neuroCombat(dat=data.T, covars=covariates, 
                                batch_col='site',
                                continuous_cols='age',
                                categorical_cols = ['sex', 'group'])['data']
        
        return pd.DataFrame(data_combat.T, index=data.index, columns=data.columns)


## General Linear Model
class general_linear_model:
    def __init__(self, demo, biomarker):
        '''initialize linear model'''

        self.biomarker = biomarker
        self.demo = demo.loc[biomarker.index]
    
    def sex_effects(self, df_name, sex_col='sex', icv_col='icv'):
        '''this is to compare the sex effects irrespective of the diagnosis'''
        # confirm the datatype
        if df_name.startswith('ctx'):
            cortical_thickness = True
            volume = False
        elif df_name.startswith(('sv', 'sa')):
            cortical_thickness = False
            volume = True
        else:
            raise ValueError('unknown dataframe prefix. Expected ctx, sv or sa')
        
        # prepare covariates
        # sex
        sex = self.demo[sex_col].to_numpy().astype(float)
        sex[sex == 1] = 1
        sex[sex == 2] = 0
        term_sex = FixedEffect(sex, 'sex')

        # icv
        icv = self.demo[icv_col].to_numpy()
        term_icv = FixedEffect(icv, 'icv')

        # model setup 
        if cortical_thickness:
            model = term_sex 
            slm_model = SLM(model, contrast=model.sex)
        else:  
            model = term_sex + term_icv
            slm_model = SLM(model, contrast=model.sex)

        # model fit
        slm_model.fit(self.biomarker)

        # stats
        t_val = slm_model.t
        p_val = (ss.t.sf(abs(slm_model.t), slm_model.df) * 2).flatten()
        rejected, q_val = fdrcorrection(p_val)
        cohen_val = slm_model.t / math.sqrt(self.biomarker.shape[0])

        return {'t_values': t_val, 'q_values': q_val, 'cohen_d': cohen_val}

    def diagnosis_effects_across_sexes(self, df_name, group_col='group', age_col='age', sex_col='sex', icv_col='icv', gender='male'):
        '''this is to compare the diagnosis effects in male or female'''
        
        # confirm the datatype
        if df_name.startswith('ctx'):
            cortical_thickness = True
            volume = False
        elif df_name.startswith(('sv', 'sa')):
            cortical_thickness = False
            volume = True
        else:
            raise ValueError('unknown dataframe prefix. Expected ctx, sv or sa')

        # confirm the epilepsy type
        if '_allepi' in df_name:
            epi_type = 'allepi'
        elif '_tle' in df_name:
            epi_type = 'tle'
        elif '_ige' in df_name:
            epi_type = 'ige'
        else:
            epi_type = 'allepi'                                               

        # filter the indices according to epilepsy type
        if gender == 'male':
            filtered_indices = self.demo[self.demo[sex_col] == 1].index
        elif gender == 'female':
            filtered_indices = self.demo[self.demo[sex_col] == 2].index
        else:
            filtered_indices = self.demo.index

        # sort out the demo and biomarker dataframe
        filtered_demo = self.demo.loc[filtered_indices]
        filtered_df = self.biomarker.loc[filtered_indices]

        age = filtered_demo[age_col].to_numpy()
        term_age = FixedEffect(age, 'age')

        icv = filtered_demo[icv_col].to_numpy()
        term_icv = FixedEffect(icv, 'icv')

        group = filtered_demo[group_col].to_numpy().astype(float)
        group[group == 1] = 1
        group[group == 0] = 0
        term_group = FixedEffect(group, 'group')

        if cortical_thickness:
            model = term_age + term_group
            slm_model = SLM(model, contrast=model.group)
        else:  
            model = term_age + term_group + term_icv
            slm_model = SLM(model, contrast=model.group)

        slm_model.fit(filtered_df)

        t_val = slm_model.t
        p_val = (ss.t.sf(abs(slm_model.t), slm_model.df) * 2).flatten()
        rejected, q_val = fdrcorrection(p_val)
        cohen_val = slm_model.t / math.sqrt(filtered_df.shape[0])

        return {'t_values': t_val, 'q_values': q_val, 'cohen_d': cohen_val, 'epilepsy_type': epi_type}
        
    def biomarker_comparison(self, df_name, comparison_type='group', 
                      age_col='age', sex_col='sex', icv_col='icv', group_col='group'):
        """
        Parameters:
        df: Dataframe to be analyzed
        df_name: Name of the dataframe (e.g., 'ctx_tle', 'sa_ige', 'sv_allepi', etc.)
        comparison_type: Type of comparison ('case_control', 'sex_in_hc', 'sex_in_epi', 'sex_group_interaction')
        age_col, sex_col, icv_col, group_col: Corresponding column names
        """

        if df_name.startswith('ctx'):
            cortical_thickness = True
            volume = False
        elif df_name.startswith(('sv', 'sa')):
            cortical_thickness = False
            volume = True
        else:
            raise ValueError('unknown dataframe prefix. Expected ctx, sv or sa')
        
        if '_allepi' in df_name:
            epi_type = 'allepi'
        elif '_tle' in df_name:
            epi_type = 'tle'
        elif '_ige' in df_name:
            epi_type = 'ige'
        else:
            epi_type = 'allepi'       
        
        if comparison_type == 'sex_in_hc':
            filtered_indices = self.demo[self.demo[group_col] == 0].index
        elif comparison_type == 'sex_in_epi':
            filtered_indices = self.demo[self.demo[group_col] == 1].index
        else:
            filtered_indices = self.demo.index

        filtered_demo = self.demo.loc[filtered_indices]
        filtered_df = self.biomarker.loc[filtered_indices]

        age = filtered_demo[age_col].to_numpy()
        term_age = FixedEffect(age, 'age')

        sex = filtered_demo[sex_col].to_numpy().astype(float)
        sex[sex == 1] = 1
        sex[sex == 2] = 0
        term_sex = FixedEffect(sex, 'sex')

        icv = filtered_demo[icv_col].to_numpy()
        term_icv = FixedEffect(icv, 'icv')

        if comparison_type in ['case_control', 'sex_group_interaction']:
            group = filtered_demo[group_col].to_numpy().astype(float)
            group[group == 1] = 1
            group[group == 0] = 0
            term_group = FixedEffect(group, 'group')
            
            # sex group interaction
            if comparison_type == 'sex_group_interaction':
                interact = sex * group
                term_interact = FixedEffect(interact, 'interact') 

        #  comparison type selection
        if comparison_type == 'case_control':
            if cortical_thickness == True:
                model = term_sex + term_age + term_group
                slm_model = SLM(model, contrast=model.group)
            else:
                model = term_sex + term_age + term_group + term_icv
                slm_model = SLM(model, contrast=model.group)
        
        elif comparison_type == 'sex_in_hc':
            if cortical_thickness:
                model = term_sex + term_age
                slm_model = SLM(model, contrast=model.sex)
            else:  
                model = term_sex + term_age + term_icv
                slm_model = SLM(model, contrast=model.sex)      
        
        elif comparison_type == 'sex_in_epi':
            if cortical_thickness:
                model = term_sex + term_age
                slm_model = SLM(model, contrast=model.sex)
            else:  
                model = term_sex + term_age + term_icv
                slm_model = SLM(model, contrast=model.sex)

        elif comparison_type == 'sex_group_interaction':
            if cortical_thickness:
                model = term_sex + term_age + term_group + term_interact
                slm_model = SLM(model, contrast=model.interact)
            else:  
                model = term_sex + term_age + term_group + term_interact + term_icv
                slm_model = SLM(model, contrast=model.interact)

        else:
            raise ValueError("Unknown comparison type, expected: 'case_control', 'sex_in_hc', 'sex_in_epi', 'sex_group_interaction'")
        
        slm_model.fit(filtered_df)

        t_val = slm_model.t
        p_val = (ss.t.sf(abs(slm_model.t), slm_model.df) * 2).flatten()
        rejected, q_val = fdrcorrection(p_val)
        cohen_val = slm_model.t / math.sqrt(filtered_df.shape[0])

        return {'t_values': t_val, 'q_values': q_val, 'cohen_d': cohen_val, 'epilepsy_type': epi_type}      

    def clinical_correlate(self, df_name, comparison_type='duration', durill='durill', ao='ao', age='age',
                            sex='sex', icv='icv'):
        """      
        Parameters:
        df: Dataframe to be analyzed
        df_name: Name of the dataframe (e.g., 'ctx_tle_durill', 'sa_ige_durill', 'sv_allepi_durill', etc.)
        comparison_type: Type of comparison ('duration', 'duration_sex_interaction', 'ao', 'ao_sex_interaction')
        age_col, sex_col, icv_col, group_col: Corresponding column names
        """

        if df_name.startswith('ctx'):
            cortical_thickness = True
            volume = False
        elif df_name.startswith(('sv', 'sa')):
            cortical_thickness = False
            volume = True
        else:
            raise ValueError('unknown dataframe prefix. Expected ctx, sv or sa')

        if '_allepi' in df_name:
            epi_type = 'allepi'
        elif '_tle' in df_name:
            epi_type = 'tle'
        elif '_ige' in df_name:
            epi_type = 'ige'
        else:
            epi_type = 'allepi'       


        sex = self.demo[sex].copy().to_numpy().astype(float)
        sex[sex == 1] = 1
        sex[sex == 2] = 0
        term_sex = FixedEffect(sex, 'sex')

        durill = self.demo[durill].to_numpy()
        term_durill = FixedEffect(durill, 'durill')

        ao = self.demo[ao].to_numpy()
        term_ao = FixedEffect(ao, 'ao')

        age = self.demo[age].to_numpy()
        term_age = FixedEffect(age, 'age')

        icv = self.demo[icv].to_numpy()
        term_icv = FixedEffect(icv, 'icv')        
        
        interact_durill = sex * durill
        term_interact_durill = FixedEffect(interact_durill, 'interact_durill')

        interact_ao = sex * ao
        term_interact_ao = FixedEffect(interact_ao, 'interact_ao')

        interact_age = sex * age
        term_interact_age = FixedEffect(interact_age, 'interact_age')

        ## comparison type selection
        if comparison_type == 'duration':
            if cortical_thickness == True:
                model = term_sex + term_durill
                slm_model = SLM(model, contrast=model.durill)
            else:
                model = term_sex + term_durill + term_icv
                slm_model = SLM(model, contrast=model.durill)
        elif comparison_type == 'duration_sex_interaction':
            if cortical_thickness == True:
                model = term_sex + term_durill + term_interact_durill
                slm_model = SLM(model, contrast=model.interact_durill)
            else:
                model = term_sex + term_durill + term_interact_durill + term_icv
                slm_model = SLM(model, contrast=model.interact_durill)
        elif comparison_type == 'ao':
            if cortical_thickness == True:
                model = term_sex + term_ao
                slm_model = SLM(model, contrast=model.ao)   
            else:
                model = term_sex + term_ao + term_icv
                slm_model = SLM(model, contrast=model.ao)
        elif comparison_type == 'ao_sex_interaction':
            if cortical_thickness == True:
                model = term_sex + term_ao + term_interact_ao
                slm_model = SLM(model, contrast=model.interact_ao)
            else:
                model = term_sex + term_ao + term_interact_ao + term_icv
                slm_model = SLM(model, contrast=model.interact_ao)
        elif comparison_type == 'age':
            if cortical_thickness == True:
                model = term_sex + term_age
                slm_model = SLM(model, contrast=model.age)   
            else:
                model = term_sex + term_age + term_icv
                slm_model = SLM(model, contrast=model.age)
        elif comparison_type == 'age_sex_interaction':
            if cortical_thickness == True:
                model = term_sex + term_age + term_interact_age
                slm_model = SLM(model, contrast=model.interact_age)
            else:
                model = term_sex + term_age + term_interact_age + term_icv
                slm_model = SLM(model, contrast=model.interact_age)
        else:
            raise ValueError("Unknown comparison type, expected: 'duration', 'duration_sex_interaction', 'ao', 'ao_sex_interaction'")

        slm_model.fit(self.biomarker)

        t_val = slm_model.t
        p_val = (ss.t.sf(abs(slm_model.t), slm_model.df) * 2).flatten()
        rejected, q_val = fdrcorrection(p_val)
        cohen_val = slm_model.t / math.sqrt(self.biomarker.shape[0])

        return {'t_values': t_val, 'q_values': q_val, 'cohen_d': cohen_val, 'epilepsy_type': epi_type}
    
    def clinical_correlate_across_sexes(self, df_name, durill_col='durill', ao_col='ao', age_col='age',
                                        sex_col='sex', icv_col='icv', gender='male', correlate_variable='durill'):
        '''this is to compare the diagnosis effects in male or female'''

        if df_name.startswith('ctx'):
            cortical_thickness = True
            volume = False
        elif df_name.startswith(('sv', 'sa')):
            cortical_thickness = False
            volume = True
        else:
            raise ValueError('unknown dataframe prefix. Expected ctx, sv or sa')

        if '_allepi' in df_name:
            epi_type = 'allepi'
        elif '_tle' in df_name:
            epi_type = 'tle'
        elif '_ige' in df_name:
            epi_type = 'ige'
        else:
            epi_type = 'allepi'                                               

        # get male and female data separatly 
        if gender == 'male':
            filtered_indices = self.demo[self.demo[sex_col] == 1].index
        elif gender == 'female':
            filtered_indices = self.demo[self.demo[sex_col] == 2].index
        else:
            filtered_indices = self.demo.index

        filtered_demo = self.demo.loc[filtered_indices]
        filtered_df = self.biomarker.loc[filtered_indices]

        durill = filtered_demo[durill_col].to_numpy()
        term_durill = FixedEffect(durill, 'durill')

        ao = filtered_demo[ao_col].to_numpy()
        term_ao = FixedEffect(ao, 'ao')

        age = filtered_demo[age_col].to_numpy()
        term_age = FixedEffect(age, 'age')

        icv = filtered_demo[icv_col].to_numpy()
        term_icv = FixedEffect(icv, 'icv')        

        if correlate_variable == 'durill':
            if cortical_thickness:
                model = term_durill
                slm_model = SLM(model, contrast=model.durill)
            else:  
                model = term_durill + term_icv
                slm_model = SLM(model, contrast=model.durill)
        elif correlate_variable == 'ao':
            if cortical_thickness:
                model = term_ao
                slm_model = SLM(model, contrast=model.ao)
            else:  
                model = term_ao + term_icv
                slm_model = SLM(model, contrast=model.ao)
        elif correlate_variable == 'age':
            if cortical_thickness:
                model = term_age
                slm_model = SLM(model, contrast=model.age)
            else:  
                model = term_age + term_icv
                slm_model = SLM(model, contrast=model.age)        
        else:
            raise ValueError("Unknown correlate_variable, expected: 'durill', 'ao', 'age'")        

        slm_model.fit(filtered_df)

        t_val = slm_model.t
        p_val = (ss.t.sf(abs(slm_model.t), slm_model.df) * 2).flatten()
        rejected, q_val = fdrcorrection(p_val)
        cohen_val = slm_model.t / math.sqrt(filtered_df.shape[0])

        return {'t_values': t_val, 'q_values': q_val, 'cohen_d': cohen_val, 'epilepsy_type': epi_type}
