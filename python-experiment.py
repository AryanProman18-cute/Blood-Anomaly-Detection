import sys
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder
import sklearn


# 1 SETUP & GLOBAL ENCODER
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
BLOOD_GROUP_PROBS = [0.22, 0.015, 0.32, 0.02, 0.07, 0.005, 0.33, 0.02]
BLOOD_GROUP_PROBS = [p / sum(BLOOD_GROUP_PROBS) for p in BLOOD_GROUP_PROBS]
COMPONENTS = ['Whole Blood', 'Packed Red Blood Cells', 'Fresh Frozen Plasma', 'Platelets']
SHELF_LIFE_DAYS = {'Whole Blood': 35, 'Packed Red Blood Cells': 42, 'Fresh Frozen Plasma': 365, 'Platelets': 5}
COMPONENT_PROBS = [0.20, 0.45, 0.20, 0.15]
CLASSES = ['MISSING_FIELDS', 'DUPLICATE_RECORDS', 'CONFLICTING_RECORDS', 'IMPOSSIBLE_STATES', 'EXPIRY_VIOLATIONS', 
           'TRACEABILITY_BREAKS', 'FORMAT_INCONSISTENCIES', 'TEMPORAL_ANOMALIES', 'STATISTICAL_BURSTS']

def build_feature_encoder():
    bg_cats = BLOOD_GROUPS + ['O POS', 'o+', 'B POSITIVE', '']
    state_cats = ['AVAILABLE', 'ISSUED', 'TEST_DISCARD', 'EXPIRED_DISCARD', '']
    test_cats = ['PASSED', 'REACTIVE_DISCARD', '']
    encoder = OneHotEncoder(categories=[bg_cats, state_cats, test_cats], handle_unknown='ignore', sparse_output=False)
    dummy_df = pd.DataFrame({'blood_group': bg_cats, 'state': (state_cats*4)[:len(bg_cats)], 'test_status': (test_cats*5)[:len(bg_cats)]})
    encoder.fit(dummy_df)
    return encoder

GLOBAL_ENCODER = build_feature_encoder()

def extract_features(df, enc):
    f = pd.DataFrame()
    f['donor_weight'] = pd.to_numeric(df['donor_weight'], errors='coerce').fillna(0).astype(float)
    f['donor_hb'] = pd.to_numeric(df['donor_hb'], errors='coerce').fillna(0).astype(float)
    f['quantity_issued'] = pd.to_numeric(df['quantity_issued'], errors='coerce').fillna(0).astype(float)
    l_times = []
    for _, r in df.iterrows():
        try:
            l_times.append((datetime.strptime(str(r['issue_timestamp']).strip(), '%Y-%m-%d %H:%M:%S') - 
                            datetime.strptime(str(r['collection_timestamp']).strip(), '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600.0)
        except: l_times.append(-1.0)
    f['lead_time'] = l_times
    cat_encoded = enc.transform(df[['blood_group', 'state', 'test_status']].fillna('').astype(str))
    return pd.concat([f, pd.DataFrame(cat_encoded, index=df.index)], axis=1).values


# 2 DATA GENERATION & SAFE ANOMALY INJECTION

def generate_clean_dataset(n_total=2000, start_date=datetime(2026, 1, 1), seed=42):
    np.random.seed(seed)
    random.seed(seed)
    records, donors = [], {}
    for d_idx in range(1, 501):
        donor_id = f"DNR-{d_idx:04d}"
        donors[donor_id] = {'donor_id': donor_id, 'blood_group': np.random.choice(BLOOD_GROUPS, p=BLOOD_GROUP_PROBS), 
                            'age': int(np.random.randint(18, 60)), 'weight_kg': float(np.round(np.random.uniform(50.0, 95.0), 1)), 
                            'hemoglobin_g_dl': float(np.round(np.random.uniform(12.5, 17.0), 1))}
    for i in range(n_total):
        donor = donors[random.choice(list(donors.keys()))]
        c_dt = start_date + timedelta(days=int(np.random.uniform(0, 90)), hours=int(np.random.uniform(8, 18)))
        comp = np.random.choice(COMPONENTS, p=COMPONENT_PROBS)
        expiry_dt = c_dt + timedelta(days=SHELF_LIFE_DAYS[comp])
        t_dt = c_dt + timedelta(hours=int(np.random.uniform(4, 24)))
        t_status = 'PASSED' if random.random() > 0.03 else 'REACTIVE_DISCARD'
        
        if t_status == 'PASSED':
            if random.random() < 0.82:
                issue_dt, state, recipient_id, issue_qty = t_dt + timedelta(days=np.random.uniform(0.5, min(SHELF_LIFE_DAYS[comp] * 0.75, 25))), 'ISSUED', f"RCP-{random.randint(1000, 9999)}", 1
            else:
                issue_dt, recipient_id, issue_qty, state = None, None, 0, 'AVAILABLE' if (start_date + timedelta(days=92)) < expiry_dt else 'EXPIRED_DISCARD'
        else:
            issue_dt, recipient_id, issue_qty, state = None, None, 0, 'TEST_DISCARD'
            
        records.append({'record_id': f"REC-{i+1:05d}", 'unit_id': f"UNT-2026-{i+1:05d}", 'donor_id': donor['donor_id'],
                        'blood_group': donor['blood_group'], 'donor_weight': donor['weight_kg'], 'donor_hb': donor['hemoglobin_g_dl'],
                        'component': comp, 'collection_timestamp': c_dt.strftime('%Y-%m-%d %H:%M:%S'),
                        'test_timestamp': t_dt.strftime('%Y-%m-%d %H:%M:%S'), 'test_status': t_status, 
                        'storage_location': "FRZ-A" if comp == 'Fresh Frozen Plasma' else "REF-1",
                        'expiry_timestamp': expiry_dt.strftime('%Y-%m-%d %H:%M:%S'), 'state': state,
                        'issue_timestamp': issue_dt.strftime('%Y-%m-%d %H:%M:%S') if issue_dt else '',
                        'recipient_id': recipient_id if recipient_id else '', 'quantity_issued': issue_qty,
                        'is_anomaly': 0, 'anomaly_class': 'CLEAN'})
    return pd.DataFrame(records), donors

def inject_anomalies(df_clean, anomaly_ratio=0.15, seed=42, mad_magnitude='extreme'):
    np.random.seed(seed)
    random.seed(seed)
    df = df_clean.copy()
    n_anomalies = int(len(df) * anomaly_ratio)
    
    anomaly_indices = np.random.choice(len(df), size=n_anomalies, replace=False)
    clean_indices_set = set(range(len(df))) - set(anomaly_indices)
    splits = np.array_split(anomaly_indices, len(CLASSES))
    
    for cls_name, idx_group in zip(CLASSES, splits):
        for idx in idx_group:
            df.at[idx, 'is_anomaly'], df.at[idx, 'anomaly_class'] = 1, cls_name
            if cls_name == 'MISSING_FIELDS': df.at[idx, random.choice(['blood_group', 'collection_timestamp', 'donor_id', 'test_status'])] = ''
            elif cls_name == 'DUPLICATE_RECORDS':
                # Force picking a clean record strictly preceding the anomaly index
                clean_before = [c for c in clean_indices_set if c < idx]
                while not clean_before:
                    idx = random.choice(idx_group) 
                    clean_before = [c for c in clean_indices_set if c < idx]
                df.at[idx, 'unit_id'] = df.at[random.choice(clean_before), 'unit_id']
            elif cls_name == 'CONFLICTING_RECORDS': df.at[idx, 'blood_group'] = 'AB-' if df.at[idx, 'blood_group'] == 'O+' else ('O+' if df.at[idx, 'blood_group'] != 'O+' else 'B+')
            elif cls_name == 'IMPOSSIBLE_STATES':
                df.at[idx, 'test_status'], df.at[idx, 'state'], df.at[idx, 'quantity_issued'], df.at[idx, 'recipient_id'], df.at[idx, 'issue_timestamp'] = 'REACTIVE_DISCARD', 'ISSUED', 1, 'RCP-9999', df.at[idx, 'test_timestamp']
            elif cls_name == 'EXPIRY_VIOLATIONS':
                exp_dt = datetime.strptime(df.at[idx, 'expiry_timestamp'], '%Y-%m-%d %H:%M:%S')
                df.at[idx, 'issue_timestamp'], df.at[idx, 'state'], df.at[idx, 'quantity_issued'], df.at[idx, 'recipient_id'] = (exp_dt + timedelta(days=random.randint(4, 18))).strftime('%Y-%m-%d %H:%M:%S'), 'ISSUED', 1, 'RCP-1111'
            elif cls_name == 'TRACEABILITY_BREAKS': df.at[idx, 'donor_id'] = 'DNR-9999'
            elif cls_name == 'FORMAT_INCONSISTENCIES': df.at[idx, 'blood_group'] = random.choice(['O POS', 'o+', 'B POSITIVE'])
            elif cls_name == 'TEMPORAL_ANOMALIES':
                c_dt = datetime.strptime(df.at[idx, 'collection_timestamp'], '%Y-%m-%d %H:%M:%S')
                df.at[idx, 'issue_timestamp'], df.at[idx, 'state'], df.at[idx, 'quantity_issued'] = (c_dt - timedelta(hours=random.randint(12, 48))).strftime('%Y-%m-%d %H:%M:%S'), 'ISSUED', 1
            elif cls_name == 'STATISTICAL_BURSTS':
                if mad_magnitude == 'extreme': df.at[idx, 'donor_weight'], df.at[idx, 'donor_hb'] = 188.5, 24.8
                elif mad_magnitude == 'moderate': df.at[idx, 'donor_weight'], df.at[idx, 'donor_hb'] = 115.0, 19.5
                elif mad_magnitude == 'subtle': df.at[idx, 'donor_weight'], df.at[idx, 'donor_hb'] = 98.0, 17.5
    return df


# 3 DETECTORS

class RuleBasedValidator:
    def __init__(self, donors_dict): self.donors_dict = donors_dict
    def detect(self, df):
        anomalies, seen_unit_ids = np.zeros(len(df), dtype=int), {}
        for idx, row in df.iterrows():
            flagged = False
            if any(str(row[f]).strip() in ['', 'nan', 'None'] for f in ['record_id', 'unit_id', 'donor_id', 'blood_group', 'component', 'collection_timestamp', 'test_status', 'state']): flagged = True
            if str(row['blood_group']).strip() not in set(BLOOD_GROUPS): flagged = True
            if str(row['state']).strip() not in {'AVAILABLE', 'ISSUED', 'TEST_DISCARD', 'EXPIRED_DISCARD'}: flagged = True
            
            u_id = str(row['unit_id']).strip()
            if u_id != '':
                if u_id in seen_unit_ids: flagged = True
                else: seen_unit_ids[u_id] = idx
            d_id = str(row['donor_id']).strip()
            if d_id != '' and d_id not in self.donors_dict: flagged = True
            elif d_id in self.donors_dict and str(row['blood_group']).strip() in set(BLOOD_GROUPS):
                if str(row['blood_group']).strip() != self.donors_dict[d_id]['blood_group']: flagged = True
            if row['state'] == 'ISSUED' and (str(row['recipient_id']).strip() == '' or row['quantity_issued'] != 1 or row['test_status'] != 'PASSED'): flagged = True
            try:
                c_dt = datetime.strptime(str(row['collection_timestamp']).strip(), '%Y-%m-%d %H:%M:%S')
                e_dt = datetime.strptime(str(row['expiry_timestamp']).strip(), '%Y-%m-%d %H:%M:%S')
                if str(row['issue_timestamp']).strip() != '':
                    i_dt = datetime.strptime(str(row['issue_timestamp']).strip(), '%Y-%m-%d %H:%M:%S')
                    if i_dt < c_dt or i_dt > e_dt: flagged = True
            except: pass
            if flagged: anomalies[idx] = 1
        return anomalies

class StatisticalAnomalyDetector:
    def detect(self, df):
        anomalies, w, hb = np.zeros(len(df), dtype=int), pd.to_numeric(df['donor_weight'], errors='coerce').values, pd.to_numeric(df['donor_hb'], errors='coerce').values
        def mod_z(arr):
            med, mad = np.nanmedian(arr), np.nanmedian(np.abs(arr - np.nanmedian(arr)))
            return 0.6745 * np.abs(arr - med) / (mad if mad > 0 else 1e-6)
        mz_w, mz_hb = mod_z(w), mod_z(hb)
        for idx in range(len(df)):
            if mz_w[idx] > 3.5 or mz_hb[idx] > 3.5: anomalies[idx] = 1
        return anomalies

def calc_metrics(y_true, y_pred):
    tp, fp = int(np.sum((y_true == 1) & (y_pred == 1))), int(np.sum((y_true == 0) & (y_pred == 1)))
    fn, tn = int(np.sum((y_true == 1) & (y_pred == 0))), int(np.sum((y_true == 0) & (y_pred == 0)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tp, fp, fn, tn, p, r, f1, fpr


# 4 EXECUTION & CSV EXPORT

if __name__ == "__main__":
    print(f"\n--- SCIENTIFIC AUDIT VERIFICATION ---")
    print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__} | Pandas: {pd.__version__} | sklearn: {sklearn.__version__}")

    # Primary OOS Evaluation (Train: 42, Test: 100)
    df_clean_42, d_dict_42 = generate_clean_dataset(seed=42)
    df_train = inject_anomalies(df_clean_42, seed=42)
    X_train = extract_features(df_train, GLOBAL_ENCODER)
    
    df_clean_100, d_dict_100 = generate_clean_dataset(seed=100)
    df_test = inject_anomalies(df_clean_100, seed=100)
    X_test = extract_features(df_test, GLOBAL_ENCODER)
    y_test = df_test['is_anomaly'].values
    
    # predict             :D
    iso_model = IsolationForest(contamination=0.15, random_state=42).fit(X_train)
    lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.15, novelty=True).fit(X_train)
    
    p_rule = RuleBasedValidator(d_dict_100).detect(df_test)
    p_mad = StatisticalAnomalyDetector().detect(df_test)
    p_hyb = np.bitwise_or(p_rule, p_mad)
    p_if_oos = np.where(iso_model.predict(X_test) == -1, 1, 0)
    p_lof_oos = np.where(lof_model.predict(X_test) == -1, 1, 0)
    
    res_data = []
    for name, p in [('Rule', p_rule), ('MAD', p_mad), ('IF_OOS', p_if_oos), ('LOF_OOS', p_lof_oos), ('Hybrid', p_hyb)]:
        tp, fp, fn, tn, prec, rec, f1, fpr = calc_metrics(y_test, p)
        res_data.append([name, tp, fp, fn, tn, f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{fpr:.3f}"])
    pd.DataFrame(res_data, columns=['Metric','TP','FP','FN','TN','Precision','Recall','F1','FPR']).to_csv('metrics_overall_oos.csv', index=False)
    
    sens_data = []
    for c in [0.05, 0.10, 0.15, 0.20]:
        p_c = np.where(IsolationForest(contamination=c, random_state=42).fit(X_train).predict(X_test) == -1, 1, 0)
        tp, fp, fn, tn, prec, rec, f1, fpr = calc_metrics(y_test, p_c)
        sens_data.append([c, tp, fp, fn, tn, f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{fpr:.3f}"])
    pd.DataFrame(sens_data, columns=['Contam', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1', 'FPR']).to_csv('metrics_sensitivity_oos.csv', index=False)

    pc_data = []
    for cls in CLASSES:
        mask = (df_test['anomaly_class'] == cls)
        tot = int(mask.sum())
        r_rec, m_rec, if_rec, lof_rec = [np.sum(mask & (p == 1))/tot for p in [p_rule, p_mad, p_if_oos, p_lof_oos]]
        pc_data.append([cls, tot, f"{r_rec*100:.1f}%", f"{m_rec*100:.1f}%", f"{if_rec*100:.1f}%", f"{lof_rec*100:.1f}%"])
    pd.DataFrame(pc_data, columns=['Anomaly_Class', 'Count', 'Rule_Recall', 'MAD_Recall', 'IF_OOS_Recall', 'LOF_OOS_Recall']).to_csv('metrics_per_class_oos.csv', index=False)
    
    mad_data = []
    for mag in ['extreme', 'moderate', 'subtle']:
        df_inj = inject_anomalies(df_clean_100, seed=100, mad_magnitude=mag)
        p_m = StatisticalAnomalyDetector().detect(df_inj)
        overall = calc_metrics(df_inj['is_anomaly'].values, p_m)
        burst_rec = np.sum((df_inj['anomaly_class'] == 'STATISTICAL_BURSTS') & (p_m == 1)) / 33.0
        mad_data.append([mag, overall[0], overall[5], overall[6], f"{burst_rec*100:.1f}%"])
    pd.DataFrame(mad_data, columns=['Magnitude', 'TP', 'Recall', 'F1', 'Burst_Recall']).to_csv('metrics_mad_sensitivity.csv', index=False)

    df_test.to_csv("dataset_test_seed100.csv", index=False)
    print("[✔] Experiment successfully executed. All OOS CSV artifacts generated.")
