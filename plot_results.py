
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Add project root to sys.path
sys.path.append('/root/autodl-tmp/iTransformer-main')

from data_provider.data_loader import Dataset_Custom

import matplotlib.font_manager as fm

import joblib

def plot_results():
    # Configuration (Match your training settings)
    root_path = './data/'
    data_path = 'part_5two_1min.csv'
    seq_len = 96
    label_len = 48
    pred_len = 96
    target = 'MM256,MM264'
    
    # Configure Font for Chinese
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    my_font = fm.FontProperties(fname=font_path, size=10.5)
    
    # Results path
    setting = 'part_5two_1min_pca_M_MM256_MM264_Robust_ScaledTargets_Transformer_custom_M_ft96_sl48_ll96_pl256_dm8_nh2_el1_dl512_df1_fctimeF_ebTrue_dtExp_PCA_Transformer_ScaledTargets_projection_0'
    results_dir = os.path.join('./results', setting)
    
    print(f"Loading results from {results_dir}")
    preds = np.load(os.path.join(results_dir, 'pred.npy'))
    trues = np.load(os.path.join(results_dir, 'true.npy'))
    
    print(f"Preds shape: {preds.shape}")
    print(f"Trues shape: {trues.shape}")
    
    # Load Data to get Timestamps
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    
    # Assuming default split ratios from run.py if not specified
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2 # implied
    
    num_train = int(len(df_raw) * train_ratio)
    num_vali = int(len(df_raw) * val_ratio)
    num_test = len(df_raw) - num_train - num_vali
    
    # Test set borders
    seq_len = 96
    border1 = len(df_raw) - num_test - seq_len
    border2 = len(df_raw)
    
    # Get the time column
    date_col = 'datetime' if 'datetime' in df_raw.columns else 'date'
    df_raw[date_col] = pd.to_datetime(df_raw[date_col])

    # Re-fit Scaler on Training Data (Targets Only)
    from sklearn.preprocessing import StandardScaler
    scaler_target = StandardScaler()
    
    # Fit on training part of target columns
    # Note: Dataset_Custom fits on [0 : num_train]
    train_slice = df_raw.iloc[0:num_train]
    target_cols = target.split(',')
    scaler_target.fit(train_slice[target_cols].values)
    print("Fitted scaler on training data targets.")
    
    # Inverse Transform
    # preds and trues are [samples, pred_len, variates]
    samples, p_len, variates = preds.shape
    preds_reshaped = preds.reshape(-1, variates)
    trues_reshaped = trues.reshape(-1, variates)
    
    print("Inverse transforming...")
    preds_inv = scaler_target.inverse_transform(preds_reshaped).reshape(samples, p_len, variates)
    trues_inv = scaler_target.inverse_transform(trues_reshaped).reshape(samples, p_len, variates)
    
    preds = preds_inv
    trues = trues_inv
    
    # The test loader iterates from index 0 to len(dataset)
    # The dataset __getitem__ returns data starting at index s_begin
    # s_begin goes from 0 to len(data_x) - seq_len - pred_len + 1
    # The target (y) starts at r_begin = s_end - label_len = index + seq_len - label_len
    # The prediction target corresponds to y[:, -pred_len:, :]
    # So the first prediction (index=0) corresponds to time steps:
    #   start: border1 + seq_len
    #   end: border1 + seq_len + pred_len
    
    # We want to plot the 1-step ahead forecast for the whole test set.
    # The 1st step of the 0-th prediction corresponds to time: border1 + seq_len
    # The 1st step of the i-th prediction corresponds to time: border1 + seq_len + i
    
    test_start_idx = border1 + seq_len
    
    # Check if lengths match
    # preds shape[0] should be roughly num_test - seq_len - pred_len + 1
    expected_len = (border2 - border1) - seq_len - pred_len + 1
    print(f"Expected test samples: {expected_len}")
    print(f"Actual preds samples: {preds.shape[0]}")
    
    # Extract timestamps for the 1-step ahead prediction
    # We use the first step of each prediction (preds[:, 0, :])
    # This corresponds to time points from test_start_idx to test_start_idx + len(preds)
    
    timestamps = df_raw[date_col].iloc[test_start_idx : test_start_idx + len(preds)]
    
    # Plotting
    targets = target.split(',')
    
    for i, target_name in enumerate(targets):
        plt.figure(figsize=(7.5, 5))
        
        # Plot 1-step ahead forecast
        # preds shape: [samples, pred_len, variates]
        # We take the first step: preds[:, 0, i]
        
        pred_series = preds[:, 0, i]
        true_series = trues[:, 0, i]
        
        # Slice data to 800-1600
        start_idx = 800
        end_idx = 1600
        # Ensure we don't go out of bounds
        real_end_idx = min(end_idx, len(pred_series))
        
        pred_slice = pred_series[start_idx:real_end_idx]
        true_slice = true_series[start_idx:real_end_idx]
        
        # Use sample index for x-axis (reset to 0-800)
        x_axis = range(0, len(pred_slice))
        
        plt.plot(x_axis, true_slice, label='真实值', linewidth=1)
        plt.plot(x_axis, pred_slice, label='预测值', linewidth=1, alpha=0.8)
        
        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Ensure left spine is at x=0
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', min(true_slice.min(), pred_slice.min()) - 0.01)) # Adjust bottom spine if needed, but usually 0 or min val
        # Actually, user said "0那块坐标线不要超出去", usually means the corner. 
        # Let's just stick to standard axes but with direction='in'
        
        # Reset bottom spine to standard (axes usually start at min value or 0)
        # But user wants "纵坐标这条线要放在横坐标0点这个位置" -> left spine at x=0.
        # And "0那块坐标线不要超出去" -> maybe means ticks don't stick out.
        
        ax.spines['left'].set_position(('data', 0))
        # Ensure x-axis starts at 0
        plt.xlim(left=0)
        
        # Ticks point inwards
        ax.tick_params(axis='both', which='both', direction='in')
        
        # Set tick font properties
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(my_font)
        
        plt.xlabel('样本序号', fontproperties=my_font)
        
        if target_name == 'MM256':
            plt.ylabel('MM256回风顺槽中部瓦斯浓度/%', fontproperties=my_font)
        elif target_name == 'MM264':
            plt.ylabel('MM264上隅角瓦斯浓度/%', fontproperties=my_font)
        else:
            plt.ylabel('浓度', fontproperties=my_font)

        plt.legend(prop=my_font)
        # plt.grid(True, alpha=0.3) # Removed grid as requested
        
        if target_name == 'MM256':
            save_path = 'pt1.png'
        elif target_name == 'MM264':
            save_path = 'pt2.png'
        else:
            save_path = f'prediction_plot_robust_zoomed_zh_{target_name}.png'
            
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()

if __name__ == "__main__":
    plot_results()
