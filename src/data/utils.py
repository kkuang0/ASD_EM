import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from typing import Tuple, List, Optional, Union
import numpy as np
import os


def create_splits_from_existing(train_val_csv_path: str, 
                               test_csv_path: str,
                               n_splits: int = 5, 
                               seed: int = 42) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    """
    Create stratified group k-fold splits using existing train-val and test CSVs
    
    Args:
        train_val_csv_path: Path to training/validation metadata CSV
        test_csv_path: Path to test metadata CSV
        n_splits: Number of CV folds
        seed: Random seed
        
    Returns:
        List of (train_df, val_df) tuples and test_df
    """
    # Load existing splits
    df_trainval = pd.read_csv(train_val_csv_path)
    df_test = pd.read_csv(test_csv_path)
    
    print(f"Loaded train/val set: {len(df_trainval)} samples")
    print(f"Loaded test set: {len(df_test)} samples")
    print(f"Train/val unique patients: {df_trainval['patient_id'].nunique()}")
    print(f"Test unique patients: {df_test['patient_id'].nunique()}")
    
    # Verify no patient overlap between train/val and test
    train_patients = set(df_trainval['patient_id'].unique())
    test_patients = set(df_test['patient_id'].unique())
    overlap = train_patients.intersection(test_patients)
    
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping patients between train/val and test sets!")
        print(f"Overlapping patients: {list(overlap)[:10]}...")  # Show first 10
    else:
        print("✓ No patient overlap between train/val and test sets")
    
    # Create stratification column combining all tasks
    df_trainval['stratify_col'] = (df_trainval['pathology'].astype(str) + '_' + 
                                  df_trainval['region'].astype(str) + '_' + 
                                  df_trainval['depth'].astype(str))
    
    if "fold" in df_trainval.columns:
        print("✓ Using pre-computed folds from 'fold' column")
        folds = []
        for k in sorted(df_trainval["fold"].unique()):
            train_df = df_trainval[df_trainval["fold"] != k].copy()
            val_df   = df_trainval[df_trainval["fold"] == k].copy()
            folds.append((train_df, val_df))
            print(f"Fold {k}: Train={len(train_df)} | Val={len(val_df)}")
        return folds, df_test
    
    print("No 'fold' column found – generating multilabel patient split")
    df_with_folds = multilabel_patient_kfold(
        df_trainval, n_splits=n_splits, seed=seed, max_per_patient=4000
    )

    folds = []
    for k in range(n_splits):
        train_df = df_with_folds[df_with_folds["fold"] != k].copy()
        val_df   = df_with_folds[df_with_folds["fold"] == k].copy()
        folds.append((train_df, val_df))
        print(f"Fold {k}: Train={len(train_df)} | Val={len(val_df)}")
    
    '''
    # Create CV folds on training/validation data
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    
    try:
        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(df_trainval, df_trainval['stratify_col'], groups=df_trainval['patient_id'])
        ):
            train_df = df_trainval.iloc[train_idx].copy()
            val_df = df_trainval.iloc[val_idx].copy()
            
            # Verify no patient overlap within fold
            train_fold_patients = set(train_df['patient_id'].unique())
            val_fold_patients = set(val_df['patient_id'].unique())
            fold_overlap = train_fold_patients.intersection(val_fold_patients)
            
            if fold_overlap:
                print(f"WARNING: Fold {fold+1} has {len(fold_overlap)} overlapping patients!")
            
            print(f"Fold {fold+1}: Train={len(train_df)} ({len(train_fold_patients)} patients), "
                  f"Val={len(val_df)} ({len(val_fold_patients)} patients)")
            
            folds.append((train_df, val_df))
            
    except ValueError as e:
        print(f"Error creating stratified folds: {e}")
        print("Falling back to group-only splits (ignoring stratification)")
        folds = create_group_only_splits(df_trainval, n_splits, seed)
    '''
    return folds, df_test
    

def create_group_only_splits(df_trainval: pd.DataFrame, 
                            n_splits: int, 
                            seed: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fallback function to create splits based only on patient groups (no stratification)
    Useful when stratified splits fail due to insufficient samples per class combination
    """
    from sklearn.model_selection import GroupKFold
    
    print("Creating group-only splits (no stratification)")
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(df_trainval, groups=df_trainval['patient_id'])
    ):
        train_df = df_trainval.iloc[train_idx].copy()
        val_df = df_trainval.iloc[val_idx].copy()
        
        train_patients = len(train_df['patient_id'].unique())
        val_patients = len(val_df['patient_id'].unique())
        
        print(f"Fold {fold+1}: Train={len(train_df)} ({train_patients} patients), "
              f"Val={len(val_df)} ({val_patients} patients)")
        
        folds.append((train_df, val_df))
    
    return folds


def create_splits(csv_path: Optional[str] = None,
                 train_val_csv_path: Optional[str] = None,
                 test_csv_path: Optional[str] = None,
                 n_splits: int = 5, 
                 holdout_frac: float = 0.2, 
                 seed: int = 42) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    """
    Unified function to create splits from either single CSV or existing train/test splits
    
    Args:
        csv_path: Path to single metadata CSV (for creating new splits)
        train_val_csv_path: Path to existing training/validation CSV
        test_csv_path: Path to existing test CSV
        n_splits: Number of CV folds
        holdout_frac: Fraction for test set (only used with single CSV)
        seed: Random seed
        
    Returns:
        List of (train_df, val_df) tuples and test_df
    """
    # Check which mode to use
    if train_val_csv_path and test_csv_path:
        # Use existing splits
        if not os.path.exists(train_val_csv_path):
            raise FileNotFoundError(f"Train/val CSV not found: {train_val_csv_path}")
        if not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")
        
        return create_splits_from_existing(train_val_csv_path, test_csv_path, n_splits, seed)
    
    elif csv_path:
        # Create new splits from single CSV (original functionality)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        return create_splits_from_single_csv(csv_path, n_splits, holdout_frac, seed)
    
    else:
        raise ValueError("Must provide either 'csv_path' OR both 'train_val_csv_path' and 'test_csv_path'")


def create_splits_from_single_csv(csv_path: str, 
                                 n_splits: int = 5, 
                                 holdout_frac: float = 0.2, 
                                 seed: int = 42) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    """
    Original function: Create stratified group k-fold splits from a single CSV
    
    Args:
        csv_path: Path to metadata CSV
        n_splits: Number of CV folds
        holdout_frac: Fraction for test set
        seed: Random seed
        
    Returns:
        List of (train_df, val_df) tuples and test_df
    """
    df = pd.read_csv(csv_path)
    
    print(f"Loaded dataset: {len(df)} samples from {df['patient_id'].nunique()} patients")
    
    # Create stratification column combining all tasks
    df['stratify_col'] = (df['pathology'].astype(str) + '_' + 
                         df['region'].astype(str) + '_' + 
                         df['depth'].astype(str))
    
    # Create holdout test set grouped by patient_id
    gss = GroupShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
    train_val_idx, test_idx = next(gss.split(df, groups=df['patient_id']))
    df_trainval = df.iloc[train_val_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    print(f"Created train/val set: {len(df_trainval)} samples from {df_trainval['patient_id'].nunique()} patients")
    print(f"Created test set: {len(df_test)} samples from {df_test['patient_id'].nunique()} patients")
    
    # Create CV folds on training/validation data
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    
    try:
        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(df_trainval, df_trainval['stratify_col'], groups=df_trainval['patient_id'])
        ):
            train_df = df_trainval.iloc[train_idx].copy()
            val_df = df_trainval.iloc[val_idx].copy()
            
            print(f"Fold {fold+1}: Train={len(train_df)} ({len(train_df['patient_id'].unique())} patients), "
                  f"Val={len(val_df)} ({len(val_df['patient_id'].unique())} patients)")
            
            folds.append((train_df, val_df))
            
    except ValueError as e:
        print(f"Error creating stratified folds: {e}")
        print("Falling back to group-only splits")
        folds = create_group_only_splits(df_trainval, n_splits, seed)
    
    return folds, df_test


def analyze_dataset_distribution(df: pd.DataFrame, name: str = "Dataset") -> dict:
    """Analyze the distribution of classes in the dataset"""
    analysis = {
        'name': name,
        'total_samples': len(df),
        'unique_patients': df['patient_id'].nunique(),
        'samples_per_patient': len(df) / df['patient_id'].nunique(),
        'class_distributions': {}
    }
    
    # Analyze each task
    for col in ['pathology', 'region', 'depth']:
        if col in df.columns:
            value_counts = df[col].value_counts()
            analysis['class_distributions'][col] = {
                'counts': value_counts.to_dict(),
                'percentages': (value_counts / len(df) * 100).round(2).to_dict()
            }
    
    # Print summary
    print(f"\n{name} Analysis:")
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Unique patients: {analysis['unique_patients']}")
    print(f"  Avg samples per patient: {analysis['samples_per_patient']:.1f}")
    
    for task, distribution in analysis['class_distributions'].items():
        print(f"  {task.capitalize()} distribution:")
        for class_name, count in distribution['counts'].items():
            percentage = distribution['percentages'][class_name]
            print(f"    {class_name}: {count} ({percentage}%)")
    
    return analysis


def analyze_splits(folds: List[Tuple[pd.DataFrame, pd.DataFrame]], 
                  test_df: pd.DataFrame) -> dict:
    """Analyze the distribution across all splits"""
    print("\n" + "="*50)
    print("SPLIT ANALYSIS")
    print("="*50)
    
    # Analyze test set
    test_analysis = analyze_dataset_distribution(test_df, "Test Set")
    
    # Analyze each fold
    fold_analyses = []
    for i, (train_df, val_df) in enumerate(folds):
        print(f"\n--- Fold {i+1} ---")
        train_analysis = analyze_dataset_distribution(train_df, f"Fold {i+1} Train")
        val_analysis = analyze_dataset_distribution(val_df, f"Fold {i+1} Val")
        fold_analyses.append((train_analysis, val_analysis))
    
    return {
        'test_analysis': test_analysis,
        'fold_analyses': fold_analyses
    }


def get_class_weights(df: pd.DataFrame, tasks: List[str]) -> dict:
    """Calculate class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    weights = {}
    for task in tasks:
        if task in df.columns:
            classes = df[task].unique()
            try:
                class_weights = compute_class_weight(
                    'balanced', 
                    classes=classes, 
                    y=df[task]
                )
                weights[task] = dict(zip(classes, class_weights))
                print(f"{task.capitalize()} class weights: {weights[task]}")
            except Exception as e:
                print(f"Warning: Could not compute class weights for {task}: {e}")
                weights[task] = {cls: 1.0 for cls in classes}
    
    return weights


# Example usage functions
def load_and_analyze_existing_splits(train_val_csv: str, test_csv: str, n_splits: int = 5):
    """Convenience function to load and analyze existing splits"""
    # Create splits
    folds, test_df = create_splits_from_existing(train_val_csv, test_csv, n_splits)
    
    # Analyze distributions
    analysis = analyze_splits(folds, test_df)
    
    # Calculate class weights from combined train/val data
    train_val_df = pd.read_csv(train_val_csv)
    tasks = ['pathology', 'region', 'depth']
    class_weights = get_class_weights(train_val_df, tasks)
    
    return folds, test_df, analysis, class_weights

def multilabel_patient_kfold(
        df: pd.DataFrame,
        n_splits: int = 5,
        seed: int = 42,
        max_per_patient: int = 4000    # keep in sync with preprocessing
    ) -> pd.DataFrame:
    """
    Returns a *copy* of `df` with a new column `fold` (0 … n_splits-1).
    1. Caps tiles per patient at `max_per_patient`
    2. Builds a 7-length multihot vector  [ASD, CTRL, A25, A46, OFC, DWM, SWM]
    3. Runs MultilabelStratifiedKFold on patients
    """
    # --- 1. cap -------------------------------------------------------------
    df_cap = (df.groupby("patient_id", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max_per_patient),
                                          random_state=seed))
                .reset_index(drop=True))

    # --- 2. multihot --------------------------------------------------------
    patient_tbl = (
        df_cap.groupby("patient_id")
              .agg(
                  ASD  = ("pathology", lambda s: int("ASD"     in set(s))),
                  CTRL = ("pathology", lambda s: int("Control" in set(s))),
                  A25  = ("region"   , lambda s: int("A25" in set(s))),
                  A46  = ("region"   , lambda s: int("A46" in set(s))),
                  OFC  = ("region"   , lambda s: int("OFC" in set(s))),
                  DWM  = ("depth"    , lambda s: int("DWM" in set(s))),
                  SWM  = ("depth"    , lambda s: int("SWM" in set(s))),
              )
              .reset_index()
    )

    X_dummy = patient_tbl[["patient_id"]]          # ignored
    Y_multi = patient_tbl.drop(columns="patient_id").values

    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    patient_tbl["fold"] = -1
    for f, (_, val_idx) in enumerate(mskf.split(X_dummy, Y_multi)):
        patient_tbl.loc[val_idx, "fold"] = f

    # --- 3. broadcast back --------------------------------------------------
    df_cap = df_cap.merge(
        patient_tbl[["patient_id", "fold"]], on="patient_id", how="left"
    )
    return df_cap