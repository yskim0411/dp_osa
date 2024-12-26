def sc1_cols():
    cat_cols = ['SEX', 'PSQI_Total', 'SSS', 'ESS_Total', 'BQ_Risk', 'ISI_Total', 'PSG_M_01_Hypnotics']
    num_cols = ['AGE', 'Height_cm', 'Weight_kg', 'BMI', 'Neckcir_cm', 'PSG_M_03_SubSD_hr']
    
    return cat_cols, num_cols


def sc2_cols():
    cat_cols, num_cols = sc1_cols()
    cat2_cols = ['PSQI_05_a', 'PSQI_05_b', 'PSQI_05_c', 'PSQI_05_d', 'PSQI_05_e', 'PSQI_05_f', 
                'PSQI_05_g', 'PSQI_05_h', 'PSQI_05_i', 'PSQI_05_j', 'PSQI_06', 'PSQI_07', 'PSQI_08', 'PSQI_09',
                'ESS_01_book', 'ESS_02_tv', 'ESS_03_sitting', 'ESS_04_transport', 'ESS_05_rest', 'ESS_06_talk', 'ESS_07_meal', 'ESS_08_driving',
                'BQ_01', 'BQ_02', 'BQ_03', 'BQ_04', 'BQ_05', 'BQ_06', 'BQ_07', 'BQ_08', 'BQ_09', 'BQ_10',
                'ISI_01_a', 'ISI_01_b', 'ISI_01_c', 'ISI_02', 'ISI_03', 'ISI_04', 'ISI_05',
                'PSG_M_02_SubSL_Home', 'PSG_M_03_SubSD_Home', 
                'PSG_M_05_Alertness', 'PSG_M_06_SQ_a', 'PSG_M_06_SQ_b', 'PSG_M_06_SQ_c', 'PSG_M_06_SQ_d',
                'PSG_M_06_SQ_e', 'PSG_M_07_Dream', 'PSG_M_08_Wake']
    num2_cols = ['PSG_M_02_SubSL_min', 'PSQI_TIB', 'PSQI_SD', 'PSQI_HSE']
    
    cat_cols += cat2_cols
    num_cols += num2_cols
    
    return cat_cols, num_cols

    
def sc3_cols():
    cat_cols, num_cols = sc1_cols()
    psg_cols = ['TST_min', 'SL_min', 'REM_SL_min', 'Sleep_Eff', 'Arousal_no', 'Arousal_idx', 'REM_pct', 'N1_pct', 'N2_pct', 'N3_pct', 'WASO_pct']
    num_cols += psg_cols
    
    return cat_cols, num_cols

def sc4_cols():
    cat_cols, num_cols = sc3_cols()
    psg_cols = [ 'Arousal_resp_idx', 'Arousal_snoring_idx', 'Arousal_PLM_idx', 'Arousal_spont_idx', 'REM_sup_min', 'NREM_sup_min',
           'PLM_idx', 'LM_idx', 'Arousal_PLM_no', 'Arousal_PLM_idx_re', 'Arousal_LM_no', 'Arousal_LM_idx',
           'AI_obs', 'AI_obs_REM',	'AI_obs_NREM', 'AI_cent', 'AI_cent_REM', 'AI_cent_NREM', 'AI_mix', 'AI_mix_REM', 'AI_mix_NREM', 'HI', 'HI_REM', 'HI_NREM',
           'AHI_total', 'RDI_no', 'Lowest_SpO2', 'AHI_sup_REM', 'AHI_lat_REM', 'AHI_sup_N1', 'AHI_lat_N1', 'AHI_sup_N2', 'AHI_lat_N2', 'AHI_sup_N3', 'AHI_lat_N3', 'AHI_sup_NREM', 'AHI_lat_NREM']
    num_cols += psg_cols
    
    return cat_cols, num_cols