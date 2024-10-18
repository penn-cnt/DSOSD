# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd
from scipy.linalg import hankel
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from scipy.stats import gamma
from kneed import KneeLocator

# Imports for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .DSOSD_utils import num_wins, MovingWinClips
import warnings

class NDDmodel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NDDmodel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1,:])
        return out
    
    def __str__(self):
         return "NDD"

class NDDIso(nn.Module):
    def __init__(self, num_channels, hidden_size):
        super(NDDIso, self).__init__()
        self.num_channels = num_channels
        self.lstms = nn.ModuleList([nn.LSTM(1, hidden_size, batch_first=True) for _ in range(num_channels)])
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_channels)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_channels):
            out, _ = self.lstms[i](x[:, :, i].unsqueeze(-1))  # LSTM input shape: (batch_size, seq_len, 1)
            out = self.fcs[i](out[:, -1, :])  # FC input shape: (batch_size, hidden_size)
            outputs.append(out.unsqueeze(1))  # Add channel dimension back

        # Concatenate outputs along channel dimension
        output = torch.cat(outputs, dim=1).squeeze()  # shape: (batch_size, num_channels, 1)
        return output
    
    def __str__(self):
         return "NDDIso"
    
class NDD:
    def __init__(self, hidden_size = 10, fs = 128,
                  train_win = 12, pred_win = 1,
                  w_size = 1, w_stride = 0.5,
                  num_epochs = 100, batch_size = 'full',
                  lr = 0.01,
                  use_cuda = False):
        self.hidden_size = hidden_size
        self.train_win = train_win
        self.pred_win = pred_win
        self.w_size = w_size
        self.w_stride = w_stride
        self.fs = fs
        self.use_cuda = use_cuda
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.is_fitted = False

        if self.use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available, using CPU instead.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    
    def _fit_scaler(self, x):
        self.scaler = RobustScaler().fit(x)

    def _scaler_transform(self, x):
        col_names = x.columns
        return pd.DataFrame(self.scaler.transform(x),columns=col_names)
    
    def _prepare_segment(self, data, ret_time=False):
        data_ch = data.columns.to_list()
        data_np = data.to_numpy()
        # How many 
        j = int(self.w_size*self.fs-(self.train_win+self.pred_win)+1)
        # j = int(data_np.shape[0] - (self.train_win+self.pred_win) + 1)
        nwins = num_wins(data_np.shape[0],self.fs,self.w_size,self.w_stride)
        data_mat = torch.zeros((nwins,j,(self.train_win+self.pred_win),data_np.shape[1]))
        for k in range(len(data_ch)): # Iterating through channels
            samples = MovingWinClips(data_np[:,k],self.fs,self.w_size,self.w_stride)
            for i in range(samples.shape[0]):
                clip = samples[i,:]
                mat = torch.tensor(hankel(clip[:j],clip[-(self.train_win+self.pred_win):]))
                data_mat[i,:,:,k] = mat
        time_mat = MovingWinClips(np.arange(len(data))/self.fs,self.fs,self.w_size,self.w_stride)
        win_times = time_mat[:,0]
        data_flat = data_mat.reshape((-1,self.train_win + self.pred_win,len(data_ch)))
        input_data = data_flat[:,:-1,:].float()
        target_data = data_flat[:,-1,:].float()

        if ret_time:
            return input_data, target_data, win_times
        else:
            return input_data, target_data
    
    def _train_model(self,dataloader,criterion,optimizer):
        # Training loop
        tbar = tqdm(range(self.num_epochs),leave=False)
        for e in tbar:
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                del inputs, targets, outputs
            if e % 10 == 9:
                tbar.set_description(f"{loss.item():.4f}")
                del loss

    def _repair_data(self,outputs,X):
        nwins = num_wins(X.shape[0],self.fs,self.w_size,self.w_size)
        nchannels = X.shape[1]
        repaired = outputs.reshape((nwins,self.w_size*self.fs-(self.train_win + self.pred_win)+1,nchannels))
        return repaired

    def fit(self, X):
        input_size = X.shape[1]
        # Initialize the model
        self.model = NDDmodel(input_size, self.hidden_size)
        self.model = self.model.to(self.device)
        # Scale the training data
        self._fit_scaler(X)
        X_z = self._scaler_transform(X)

        # Prepare input and target data for the LSTM
        input_data,target_data = self._prepare_segment(X_z)

        dataset = TensorDataset(input_data, target_data)
        if self.batch_size == 'full':
            batch_size = len(dataset)
        else:
            batch_size = self.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Train the model, this will just modify the model object, no returns
        self._train_model(dataloader,criterion,optimizer)
        self.is_fitted = True

    def generate_loss(self, X):
        assert self.is_fitted, "Must fit model before running inference"
        X_z = self._scaler_transform(X)
        input_data,target_data, time_wins = self._prepare_segment(X_z,ret_time=True)
        self.time_wins = time_wins
        dataset = TensorDataset(input_data,target_data)
        if self.batch_size == 'full':
            batch_size = len(dataset)
        else:
            batch_size = self.batch_size
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        with torch.no_grad():
            self.model.eval()
            mse_distribution = []
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                mse = (outputs-targets)**2
                mse_distribution.append(mse)
                del inputs, targets, outputs, mse
        raw_mdl_outputs = torch.cat(mse_distribution).cpu().numpy()
        mdl_outs = raw_mdl_outputs.reshape((len(time_wins),-1,raw_mdl_outputs.shape[1]))
        # mdl_outs = self._repair_data(raw_mdl_outputs,X_z)
        raw_loss_mat = np.sqrt(np.mean(mdl_outs,axis=1)).T
        loss_mat = sc.ndimage.uniform_filter1d(raw_loss_mat,20,axis=1,mode='constant')
        self.feature_df = pd.DataFrame(loss_mat.T,columns = X.columns)
        return self.feature_df
    
    def _get_onset_and_spread(self,threshold,prob_chs):
        sz_clf = self.sz_prob>threshold
        seized_idxs = np.any(sz_clf,axis=1)
        sz_clf_df = pd.DataFrame(sz_clf).T
        sz_spread_idxs = sz_clf_df.rolling(window=5,closed='right').apply(lambda x: (x == 1).all())
        first_sz_idxs = sz_spread_idxs.idxmax().to_numpy() - 4
        if sum(seized_idxs) > 0:
            sz_times_arr = self.time_wins[first_sz_idxs[seized_idxs]]
            sz_times_arr -= np.min(sz_times_arr)
            sz_ch_arr = prob_chs[seized_idxs]
            sz_ch_arr = np.array([s.split("-")[0] for s in sz_ch_arr]).flatten()
        else:
            sz_ch_arr = []
            sz_times_arr = []
        return pd.DataFrame(np.array(sz_times_arr).reshape(1,-1),columns=sz_ch_arr)
             
    def predict_gamma(self, confidence_threshold = 0.95, pre_buffer = 120, post_buffer = 120):
        assert (confidence_threshold < 1) and (confidence_threshold >= 0),'Invalid confidence threshold'
        assert hasattr(self,'feature_df'), "Must generate loss before extracting channels"

        sz_prob = self.feature_df.copy()
        time_wins = self.time_wins
        prob_chs = sz_prob.columns.to_numpy()
        sz_prob = sz_prob.to_numpy().T
        onset_idx = np.argmin(np.abs((time_wins-pre_buffer)))
        offset_idx = np.argmin(np.abs(time_wins-(max(time_wins)-post_buffer)))
        shape, _, scale = gamma.fit(sz_prob.flatten(), floc=0)
        self.sz_prob = gamma.cdf(sz_prob.flatten(),a=shape,scale=scale).reshape(sz_prob.shape)[:,onset_idx:offset_idx]
        self.sz_spread = self._get_onset_and_spread(confidence_threshold,prob_chs)
        return self.sz_spread

    def predict_knee(self, pre_buffer = 120, post_buffer = 120):
        assert hasattr(self,'feature_df'), "Must generate loss before extracting channels"
        sz_prob = self.feature_df.copy()
        time_wins = self.time_wins
        prob_chs = sz_prob.columns.to_numpy()
        onset_idx = np.argmin(np.abs((time_wins-pre_buffer)))
        offset_idx = np.argmin(np.abs(time_wins-(max(time_wins)-post_buffer)))
        self.sz_prob = sz_prob.to_numpy().T[:,onset_idx:offset_idx]

        probabilities = self.sz_prob.flatten()
        thresh_sweep = np.linspace(min(probabilities),max(probabilities),2000)
        kde_model = sc.stats.gaussian_kde(probabilities,'scott')
        kde_vals = kde_model(thresh_sweep)

        # Find KDE peaks
        kde_peaks,_ = sc.signal.find_peaks(kde_vals)
        try:
            biggest_pk_idx = np.where(kde_vals[kde_peaks]>(np.mean(kde_vals)+np.std(kde_vals)))[0][-1]
        except:
            biggest_pk_idx = np.argmax(kde_vals[kde_peaks])
        if biggest_pk_idx == len(kde_peaks)-1:
            biggest_pk_idx = 0

        # Identify optimal threshold as knee between peaks
        if (len(kde_peaks) == 1) or (biggest_pk_idx == (len(kde_peaks)-1)):
            start, end = kde_peaks[biggest_pk_idx], int(kde_peaks[biggest_pk_idx] + (len(thresh_sweep)-kde_peaks[biggest_pk_idx])/4)
        else:
            start, end = kde_peaks[biggest_pk_idx], kde_peaks[biggest_pk_idx+1]

        kneedle = KneeLocator(thresh_sweep[start+10:end],kde_vals[start+10:end],
                curve='convex',direction='decreasing',interp_method='polynomial')
        knee = kneedle.knee
        self.sz_spread = self._get_onset_and_spread(knee,prob_chs)
        return self.sz_spread