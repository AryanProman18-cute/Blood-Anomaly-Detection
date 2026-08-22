import sys
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder
import sklearn

# CONSTANTS & CONFIGURATION
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
BLOOD_GROUP_PROBS = [0.22, 0.015, 0.32, 0.02, 0.07, 0.005, 0.33, 0.02]
BLOOD_GROUP_PROBS = [p / sum(BLOOD_GROUP_PROBS) for p in BLOOD_GROUP_PROBS]

COMPONENTS = ['Whole Blood', 'Packed Red Blood Cells', 'Fresh Frozen Plasma', 'Platelets']
SHELF_LIFE_DAYS = {'Whole Blood': 35, 'Packed Red Blood Cells': 42, 'Fresh Frozen Plasma': 365, 'Platelets': 5}
COMPONENT_PROBS = [0.20, 0.45, 0.20, 0.15]

STORAGE_LOCATIONS = ['FRZ-A', 'REF-1']

CLASSES = [
    'MISSING_FIELDS', 'DUPLICATE_RECORDS', 'CONFLICTING_RECORDS',
    'IMPOSSIBLE_STATES', 'EXPIRY_VIOLATIONS', 'TRACEABILITY_BREAKS',
    'FORMAT_INCONSISTENCIES', 'TEMPORAL_ANOMALIES', 'STATISTICAL_BURSTS'
]


# 1 ROBUST GLOBAL ENCODER 

def build_feature_encoder():
    bg_cats = BLOOD_GROUPS + ['O POS', 'o+', 'B POSITIVE', '']
    state_cats = ['AVAILABLE', 'ISSUED', 'TEST_DISCARD', 'EXPIRED_DISCARD', '']
    test_cats = ['PASSED', 'REACTIVE_DISCARD', '']
    component_cats = COMPONENTS + ['']
    storage_cats = STORAGE_LOCATIONS + ['']

    categories = [bg_cats, state_cats, test_cats, component_cats, storage_cats]

    try:
        encoder = OneHotEncoder(categories=categories, handle_unknown='ignore', sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(categories=categories, handle_unknown='ignore', sparse=False)

    dummy_df = pd.DataFrame({
        'blood_group': [bg_cats[0]],
        'state': [state_cats[0]],
        'test_status': [test_cats[0]],
        'component': [component_cats[0]],
        'storage_location': [storage_cats[0]]
    })
    encoder.fit(dummy_df)
    return encoder

GLOBAL_ENCODER = build_feature_encoder()

def extract_features(df, enc):
    """Extract numeric and one-hot encoded categorical features."""
    f = pd.DataFrame(index=df.index)
    f['donor_weight'] = pd.to_numeric(df['donor_weight'], errors='coerce').fillna(0).astype(float)
    f['donor_hb'] = pd.to_numeric(df['donor_hb'], errors='coerce').fillna(0).astype(float)
    f['quantity_issued'] = pd.to_numeric(df['quantity_issued'], errors='coerce').fillna(0).astype(float)

    l_times = []
    for _, r in df.iterrows():
        try:
            c = datetime.strptime(str(r['collection_timestamp']).strip(), '%Y-%m-%d %H:%M:%S')
            i = datetime.strptime(str(r['issue_timestamp']).strip(), '%Y-%m-%d %H:%M:%S')
            l_times.append((i - c).total_seconds() / 3600.0)
        except:
            l_times.append(-1.0)
    f['lead_time'] = l_times

    cat_df = df[['blood_group', 'state', 'test_status', 'component', 'storage_location']].fillna('').astype(str)
    cat_encoded = enc.transform(cat_df)
    cat_encoded_df = pd.DataFrame(cat_encoded, index=df.index)

    return pd.concat([f, cat_encoded_df], axis=1).values


# 2. DATA GENERATION & SAFE ANOMALY INJECTION

def generate_clean_dataset(n_total=2000, start_date=datetime(2026, 1, 1), seed=42):
    np.random.seed(seed)
    random.seed(seed)
    records, donors = [], {}

    for d_idx in range(1, 501):
        donor_id = f"DNR-{d_idx:04d}"
        donors[donor_id] = {
            'donor_id': donor_id,
            'blood_group': np.random.choice(BLOOD_GROUPS, p=BLOOD_GROUP_PROBS),
            'age': int(np.random.randint(18, 60)),
            'weight_kg': float(np.round(np.random.uniform(50.0, 95.0), 1)),
            'hemoglobin_g_dl': float(np.round(np.random.uniform(12.5, 17.0), 1))
        }

    donor_id_list = list(donors.keys())
    for i in range(n_total):
        donor = donors[random.choice(donor_id_list)]
        c_dt = start_date + timedelta(days=int(np.random.uniform(0, 90)), hours=int(np.random.uniform(8, 18)))
        comp = np.random.choice(COMPONENTS, p=COMPONENT_PROBS)
        expiry_dt = c_dt + timedelta(days=SHELF_LIFE_DAYS[comp])
        t_dt = c_dt + timedelta(hours=int(np.random.uniform(4, 24)))
        t_status = 'PASSED' if random.random() > 0.03 else 'REACTIVE_DISCARD'

        if t_status == 'PASSED':
            if random.random() < 0.82:
                issue_dt = t_dt + timedelta(days=np.random.uniform(0.5, min(SHELF_LIFE_DAYS[comp] * 0.75, 25)))
                state, recipient_id, issue_qty = 'ISSUED', f"RCP-{random.randint(1000, 9999)}", 1
            else:
                issue_dt, recipient_id, issue_qty = None, None, 0
                state = 'AVAILABLE' if (start_date + timedelta(days=92)) < expiry_dt else 'EXPIRED_DISCARD'
        else:
            issue_dt, recipient_id, issue_qty, state = None, None, 0, 'TEST_DISCARD'

        records.append({
            'record_id': f"REC-{i+1:05d}",
            'unit_id': f"UNT-2026-{i+1:05d}",
            'donor_id': donor['donor_id'],
            'blood_group': donor['blood_group'],
            'donor_weight': donor['weight_kg'],
            'donor_hb': donor['hemoglobin_g_dl'],
            'component': comp,
            'collection_timestamp': c_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'test_timestamp': t_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'test_status': t_status,
            'storage_location': "FRZ-A" if comp == 'Fresh Frozen Plasma' else "REF-1",
            'expiry_timestamp': expiry_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'state': state,
            'issue_timestamp': issue_dt.strftime('%Y-%m-%d %H:%M:%S') if issue_dt else '',
            'recipient_id': recipient_id if recipient_id else '',
            'quantity_issued': issue_qty,
            'is_anomaly': 0,
            'anomaly_class': 'CLEAN'
        })
    return pd.DataFrame(records), donors

def inject_anomalies(df_clean, anomaly_ratio=0.15, seed=42, mad_magnitude='extreme'):
    np.random.seed(seed)
    random.seed(seed)
    df = df_clean.copy()
    n_total = len(df)
    n_anomalies = int(n_total * anomaly_ratio)

    available_indices = np.arange(1, n_total)
    anomaly_indices = np.random.choice(available_indices, size=n_anomalies, replace=False)
    clean_indices_set = set(range(n_total)) - set(anomaly_indices)

    splits = np.array_split(anomaly_indices, len(CLASSES))

    for cls_name, idx_group in zip(CLASSES, splits):
        for idx in idx_group:
            df.at[idx, 'is_anomaly'] = 1
            df.at[idx, 'anomaly_class'] = cls_name

            if cls_name == 'MISSING_FIELDS':
                field = random.choice([
                    'blood_group', 'collection_timestamp', 'donor_id', 'test_status',
                    'component', 'expiry_timestamp', 'state', 'unit_id'
                ])
                df.at[idx, field] = ''
            elif cls_name == 'DUPLICATE_RECORDS':
                clean_before = [c for c in clean_indices_set if c < idx]
                target_clean_idx = random.choice(clean_before)
                df.at[idx, 'unit_id'] = df.at[target_clean_idx, 'unit_id']
            elif cls_name == 'CONFLICTING_RECORDS':
                if df.at[idx, 'blood_group'] == 'O+':
                    df.at[idx, 'blood_group'] = 'AB-'
                else:
                    df.at[idx, 'blood_group'] = 'O+'
            elif cls_name == 'IMPOSSIBLE_STATES':
                df.at[idx, 'test_status'], df.at[idx, 'state'], df.at[idx, 'quantity_issued'] = 'REACTIVE_DISCARD', 'ISSUED', 1
                df.at[idx, 'recipient_id'], df.at[idx, 'issue_timestamp'] = 'RCP-9999', df.at[idx, 'test_timestamp']
            elif cls_name == 'EXPIRY_VIOLATIONS':
                exp_dt = datetime.strptime(df.at[idx, 'expiry_timestamp'], '%Y-%m-%d %H:%M:%S')
                df.at[idx, 'issue_timestamp'] = (exp_dt + timedelta(days=random.randint(4, 18))).strftime('%Y-%m-%d %H:%M:%S')
                df.at[idx, 'state'], df.at[idx, 'quantity_issued'], df.at[idx, 'recipient_id'] = 'ISSUED', 1, 'RCP-1111'
            elif cls_name == 'TRACEABILITY_BREAKS':
                df.at[idx, 'donor_id'] = 'DNR-9999'
            elif cls_name == 'FORMAT_INCONSISTENCIES':
                df.at[idx, 'blood_group'] = random.choice(['O POS', 'o+', 'B POSITIVE'])
            elif cls_name == 'TEMPORAL_ANOMALIES':
                c_dt = datetime.strptime(df.at[idx, 'collection_timestamp'], '%Y-%m-%d %H:%M:%S')
                df.at[idx, 'issue_timestamp'] = (c_dt - timedelta(hours=random.randint(12, 48))).strftime('%Y-%m-%d %H:%M:%S')
                df.at[idx, 'state'], df.at[idx, 'quantity_issued'] = 'ISSUED', 1
            elif cls_name == 'STATISTICAL_BURSTS':
                if mad_magnitude == 'extreme':
                    df.at[idx, 'donor_weight'], df.at[idx, 'donor_hb'] = 188.5, 24.8
                elif mad_magnitude == 'moderate':
                    df.at[idx, 'donor_weight'], df.at[idx, 'donor_hb'] = 115.0, 19.5
                elif mad_magnitude == 'subtle':
                    df.at[idx, 'donor_weight'], df.at[idx, 'donor_hb'] = 98.0, 17.5
    return df


# 3 DETECTORS

class RuleBasedValidator:
    def __init__(self, donors_dict):
        self.donors_dict = donors_dict

    def detect(self, df):
        anomalies, seen_unit_ids = np.zeros(len(df), dtype=int), {}
        for idx, row in df.iterrows():
            flagged = False
            # FIXED: added 'expiry_timestamp' to required fields
            if any(str(row[f]).strip() in ['', 'nan', 'None'] for f in [
                'record_id', 'unit_id', 'donor_id', 'blood_group', 'component',
                'collection_timestamp', 'test_status', 'state', 'expiry_timestamp'
            ]):
                flagged = True
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
            if row['state'] == 'ISSUED' and (str(row['recipient_id']).strip() == '' or row['quantity_issued'] != 1 or row['test_status'] != 'PASSED'):
                flagged = True
            try:
                c_dt = datetime.strptime(str(row['collection_timestamp']).strip(), '%Y-%m-%d %H:%M:%S')
                e_dt = datetime.strptime(str(row['expiry_timestamp']).strip(), '%Y-%m-%d %H:%M:%S')
                if str(row['issue_timestamp']).strip() != '':
                    i_dt = datetime.strptime(str(row['issue_timestamp']).strip(), '%Y-%m-%d %H:%M:%S')
                    if i_dt < c_dt or i_dt > e_dt: flagged = True
            except: pass
            if flagged: anomalies[idx] = 1
        return anomalies

class TrainCalibratedMADDetector:
    def __init__(self, threshold=3.5):
        self.threshold = threshold
        self.med_w, self.mad_w, self.med_hb, self.mad_hb = None, None, None, None

    def fit(self, df_train_clean):
        w = pd.to_numeric(df_train_clean['donor_weight'], errors='coerce').values
        hb = pd.to_numeric(df_train_clean['donor_hb'], errors='coerce').values

        self.med_w = float(np.nanmedian(w))
        self.mad_w = float(np.nanmedian(np.abs(w - self.med_w)))
        if self.mad_w == 0: self.mad_w = 1e-6

        self.med_hb = float(np.nanmedian(hb))
        self.mad_hb = float(np.nanmedian(np.abs(hb - self.med_hb)))
        if self.mad_hb == 0: self.mad_hb = 1e-6
        return self

    def detect(self, df_test):
        anomalies = np.zeros(len(df_test), dtype=int)
        w = pd.to_numeric(df_test['donor_weight'], errors='coerce').values
        hb = pd.to_numeric(df_test['donor_hb'], errors='coerce').values

        mz_w = 0.6745 * np.abs(w - self.med_w) / self.mad_w
        mz_hb = 0.6745 * np.abs(hb - self.med_hb) / self.mad_hb

        for idx in range(len(df_test)):
            if mz_w[idx] > self.threshold or mz_hb[idx] > self.threshold:
                anomalies[idx] = 1
        return anomalies

def calc_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * (p * r) / (p + r)) if (p + r) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return tp, fp, fn, tn, p, r, f1, fpr


# 4 EXECUTE FULL EXPERIMENTAL SUITE

if __name__ == "__main__":
    SEED_PAIRS = [
        (42, 100), (43, 101), (44, 102), (45, 103), (46, 104),
        (47, 105), (48, 106), (49, 107), (50, 108), (51, 109)
    ]

    print("=== STARTING COMPLETE AUDIT & EXPERIMENT ===")
    print(f"Environment: Python {sys.version.split()[0]} | sklearn {sklearn.__version__} | NumPy {np.__version__} | Pandas {pd.__version__}")

    results_by_seed = []
    per_class_by_seed = {cls: {'Rule': [], 'MAD': [], 'IF': [], 'LOF': []} for cls in CLASSES}
    fp_records_analysis = []

    for train_seed, test_seed in SEED_PAIRS:
        df_clean_train, d_dict_train = generate_clean_dataset(seed=train_seed)
        df_train = inject_anomalies(df_clean_train, seed=train_seed)
        X_train = extract_features(df_train, GLOBAL_ENCODER)

        # MAD fitted on CLEAN data only
        mad_detector = TrainCalibratedMADDetector(threshold=3.5).fit(df_clean_train)
        iso_model = IsolationForest(contamination=0.15, random_state=train_seed).fit(X_train)
        lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.15, novelty=True).fit(X_train)

        df_clean_test, d_dict_test = generate_clean_dataset(seed=test_seed)
        df_test = inject_anomalies(df_clean_test, seed=test_seed)
        X_test = extract_features(df_test, GLOBAL_ENCODER)
        y_test = df_test['is_anomaly'].values

        p_rule = RuleBasedValidator(d_dict_test).detect(df_test)
        p_mad = mad_detector.detect(df_test)
        p_hyb = np.bitwise_or(p_rule, p_mad)
        p_if = np.where(iso_model.predict(X_test) == -1, 1, 0)
        p_lof = np.where(lof_model.predict(X_test) == -1, 1, 0)

        for m_name, preds in [('Rule', p_rule), ('MAD', p_mad), ('IF_OOS', p_if), ('LOF_OOS', p_lof), ('Hybrid', p_hyb)]:
            tp, fp, fn, tn, p, r, f1, fpr = calc_metrics(y_test, preds)
            results_by_seed.append({
                'train_seed': train_seed, 'test_seed': test_seed, 'model': m_name,
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'Precision': p, 'Recall': r, 'F1': f1, 'FPR': fpr
            })

        for cls in CLASSES:
            mask = (df_test['anomaly_class'] == cls)
            tot = int(mask.sum())
            per_class_by_seed[cls]['Rule'].append(np.sum(mask & (p_rule == 1)) / tot)
            per_class_by_seed[cls]['MAD'].append(np.sum(mask & (p_mad == 1)) / tot)
            per_class_by_seed[cls]['IF'].append(np.sum(mask & (p_if == 1)) / tot)
            per_class_by_seed[cls]['LOF'].append(np.sum(mask & (p_lof == 1)) / tot)

        if (train_seed, test_seed) == (42, 100):
            fp_mask = (y_test == 0) & (p_if == 1)
            fp_df = df_test[fp_mask]
            clean_df = df_test[y_test == 0]
            fp_records_analysis = {
                'total_clean': len(clean_df), 'total_fp': len(fp_df),
                'bg_distribution_clean': clean_df['blood_group'].value_counts(normalize=True).to_dict(),
                'bg_distribution_fp': fp_df['blood_group'].value_counts(normalize=True).to_dict()
            }

    df_results_seeds = pd.DataFrame(results_by_seed)

    summary_list = []
    for model in ['Rule', 'MAD', 'IF_OOS', 'LOF_OOS', 'Hybrid']:
        sub = df_results_seeds[df_results_seeds['model'] == model]
        summary_list.append({
            'Model': model,
            'TP_mean': sub['TP'].mean(), 'TP_sd': sub['TP'].std(),
            'FP_mean': sub['FP'].mean(), 'FP_sd': sub['FP'].std(),
            'FN_mean': sub['FN'].mean(), 'FN_sd': sub['FN'].std(),
            'TN_mean': sub['TN'].mean(), 'TN_sd': sub['TN'].std(),
            'Precision_mean': sub['Precision'].mean(), 'Precision_sd': sub['Precision'].std(),
            'Recall_mean': sub['Recall'].mean(), 'Recall_sd': sub['Recall'].std(),
            'F1_mean': sub['F1'].mean(), 'F1_sd': sub['F1'].std(),
            'FPR_mean': sub['FPR'].mean(), 'FPR_sd': sub['FPR'].std(),
        })
    df_summary = pd.DataFrame(summary_list)

    primary_df = df_results_seeds[(df_results_seeds['train_seed'] == 42) & (df_results_seeds['test_seed'] == 100)]

    per_class_summary = []
    for cls in CLASSES:
        df_clean_100, d_dict_100 = generate_clean_dataset(seed=100)
        df_test_100 = inject_anomalies(df_clean_100, seed=100)
        mask_100 = (df_test_100['anomaly_class'] == cls)
        tot_100 = int(mask_100.sum())
        per_class_summary.append({
            'Anomaly_Class': cls, 'Count': tot_100,
            'Rule_Recall_mean': np.mean(per_class_by_seed[cls]['Rule']) * 100,
            'Rule_Recall_sd': np.std(per_class_by_seed[cls]['Rule']) * 100,
            'MAD_Recall_mean': np.mean(per_class_by_seed[cls]['MAD']) * 100,
            'MAD_Recall_sd': np.std(per_class_by_seed[cls]['MAD']) * 100,
            'IF_Recall_mean': np.mean(per_class_by_seed[cls]['IF']) * 100,
            'IF_Recall_sd': np.std(per_class_by_seed[cls]['IF']) * 100,
            'LOF_Recall_mean': np.mean(per_class_by_seed[cls]['LOF']) * 100,
            'LOF_Recall_sd': np.std(per_class_by_seed[cls]['LOF']) * 100,
        })
    df_per_class = pd.DataFrame(per_class_summary)

    if_sens_records = []
    for c in [0.05, 0.10, 0.15, 0.20]:
        c_f1s, c_recs, c_fprs, c_tps, c_fps = [], [], [], [], []
        for train_seed, test_seed in SEED_PAIRS:
            df_tr, _ = generate_clean_dataset(seed=train_seed)
            df_tr = inject_anomalies(df_tr, seed=train_seed)
            X_tr = extract_features(df_tr, GLOBAL_ENCODER)

            df_te, _ = generate_clean_dataset(seed=test_seed)
            df_te = inject_anomalies(df_te, seed=test_seed)
            X_te = extract_features(df_te, GLOBAL_ENCODER)
            y_te = df_te['is_anomaly'].values

            p_c = np.where(IsolationForest(contamination=c, random_state=train_seed).fit(X_tr).predict(X_te) == -1, 1, 0)
            tp, fp, fn, tn, p, r, f1, fpr = calc_metrics(y_te, p_c)
            c_tps.append(tp); c_fps.append(fp); c_recs.append(r); c_fprs.append(fpr); c_f1s.append(f1)

        if_sens_records.append({
            'Contamination': c, 'TP_mean': np.mean(c_tps), 'FP_mean': np.mean(c_fps),
            'Recall_mean': np.mean(c_recs), 'Recall_sd': np.std(c_recs),
            'FPR_mean': np.mean(c_fprs), 'FPR_sd': np.std(c_fprs),
            'F1_mean': np.mean(c_f1s), 'F1_sd': np.std(c_f1s)
        })
    df_if_sens = pd.DataFrame(if_sens_records)

    mad_sens_records = []
    for mag in ['extreme', 'moderate', 'subtle']:
        recs, f1s, burst_recs = [], [], []
        for train_seed, test_seed in SEED_PAIRS:
            df_clean_tr, _ = generate_clean_dataset(seed=train_seed)
            mad_det = TrainCalibratedMADDetector(threshold=3.5).fit(df_clean_tr)

            df_clean_te, _ = generate_clean_dataset(seed=test_seed)
            df_te = inject_anomalies(df_clean_te, seed=test_seed, mad_magnitude=mag)
            y_te = df_te['is_anomaly'].values

            p_m = mad_det.detect(df_te)
            tp, fp, fn, tn, p, r, f1, fpr = calc_metrics(y_te, p_m)
            recs.append(r); f1s.append(f1)

            burst_mask = (df_te['anomaly_class'] == 'STATISTICAL_BURSTS')
            burst_recs.append(np.sum(burst_mask & (p_m == 1)) / int(burst_mask.sum()))

        mad_sens_records.append({
            'Magnitude': mag, 'Overall_Recall_mean': np.mean(recs), 'Overall_F1_mean': np.mean(f1s),
            'Burst_Recall_mean': np.mean(burst_recs) * 100, 'Burst_Recall_sd': np.std(burst_recs) * 100
        })
    df_mad_sens = pd.DataFrame(mad_sens_records)

    # Save CSVs
    primary_df.to_csv('metrics_overall_oos.csv', index=False)
    df_summary.to_csv('metrics_multiseed_summary.csv', index=False)
    df_if_sens.to_csv('metrics_sensitivity_oos.csv', index=False)
    df_per_class.to_csv('metrics_per_class_oos.csv', index=False)
    df_mad_sens.to_csv('metrics_mad_sensitivity.csv', index=False)

    print("\n--- SUMMARY METRICS (MEAN ± SD ACROSS 10 SEED PAIRS) ---")
    for _, r in df_summary.iterrows():
        print(f"{r['Model'].ljust(12)}: F1={r['F1_mean']:.3f}±{r['F1_sd']:.3f} | Recall={r['Recall_mean']:.3f}±{r['Recall_sd']:.3f} | Precision={r['Precision_mean']:.3f}±{r['Precision_sd']:.3f} | FPR={r['FPR_mean']:.3f}±{r['FPR_sd']:.3f}")

    print("\n--- PRIMARY SEED PAIR (42 -> 100) METRICS ---")
    for _, r in primary_df.iterrows():
        print(f"{r['model'].ljust(12)}: TP={r['TP']} | FP={r['FP']} | FN={r['FN']} | TN={r['TN']} | F1={r['F1']:.3f} | Rec={r['Recall']:.3f} | Prec={r['Precision']:.3f} | FPR={r['FPR']:.3f}")

    print("\n--- PER-CLASS RECALL (MEAN ± SD %) ---")
    for _, r in df_per_class.iterrows():
        print(f"{r['Anomaly_Class'].ljust(25)}: Rule={r['Rule_Recall_mean']:.1f}%±{r['Rule_Recall_sd']:.1f}% | MAD={r['MAD_Recall_mean']:.1f}%±{r['MAD_Recall_sd']:.1f}% | IF={r['IF_Recall_mean']:.1f}%±{r['IF_Recall_sd']:.1f}% | LOF={r['LOF_Recall_mean']:.1f}%±{r['LOF_Recall_sd']:.1f}%")

    print("\n--- MAD MAGNITUDE SENSITIVITY ---")
    for _, r in df_mad_sens.iterrows():
        print(f"{r['Magnitude'].ljust(10)}: Burst Recall={r['Burst_Recall_mean']:.1f}%±{r['Burst_Recall_sd']:.1f}% | Overall F1={r['Overall_F1_mean']:.3f}")

    print("\n--- IF CONTAMINATION SENSITIVITY (OOS) ---")
    for _, r in df_if_sens.iterrows():
        print(f"c={r['Contamination']:.2f}: F1={r['F1_mean']:.3f}±{r['F1_sd']:.3f} | Recall={r['Recall_mean']:.3f}±{r['Recall_sd']:.3f} | FPR={r['FPR_mean']:.3f}±{r['FPR_sd']:.3f}")

    print("\n--- FALSE POSITIVE ANALYSIS (Seed 42 -> 100) ---")
    print(f"Total Clean Records: {fp_records_analysis['total_clean']} | False Positives Flagged by IF: {fp_records_analysis['total_fp']}")
    print("Clean Population Blood Group Dist:", {k: round(v, 3) for k, v in fp_records_analysis['bg_distribution_clean'].items()})
    print("False Positive Blood Group Dist:", {k: round(v, 3) for k, v in fp_records_analysis['bg_distribution_fp'].items()})

    print("\n[✔] Automated audit completed. Data exported to CSVs successfully.")
