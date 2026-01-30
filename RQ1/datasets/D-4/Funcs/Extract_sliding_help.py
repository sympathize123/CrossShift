import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Iterable, Dict, List, Union
from pandas._libs.tslibs import NaT
from sklearn.preprocessing import OrdinalEncoder
import ray
from ray.actor import ActorHandle
from glob import glob
from tqdm import tqdm

import pytz
from Funcs.Utility import *

import pygeohash as geo
from functools import reduce

from asyncio import Event
from typing import Tuple
from datetime import timedelta

#from Utility import load, log

import numpy as np
import pandas as pd
from functools import reduce
from typing import Dict
import datetime as dt
from statsmodels.tsa import stattools as st
from scipy import stats as sp
import traceback
import warnings
from collections import defaultdict
# from Funcs.preprocessing import *



def _extract_numeric_feature(_x: np.ndarray) -> Dict[str, any]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    _N = len(_x) # number of data points within the given window

    # Median
    _med = np.median(_x)

    # Minimum
    _min = np.min(_x)

    # Maximum
    _max = np.max(_x)

    # Binned Entropy (n = 10)
    _hist, _ = np.histogram(_x, bins=10, density=False)
    _bin_entr = sp.entropy(_hist)

    # Sample Mean
    _mean = np.mean(_x)

    # Sample Variance
    _var = np.var(_x, ddof=1)

    # Sample Skewness
    _skew = sp.skew(_x, bias=False)

    # Sample Kurtosis
    _kurt = sp.kurtosis(_x, bias=False)

        
    # Abs. Sum of Changes
    _asc = np.sum(np.abs(np.diff(_x)))

    # Auto-correlation: correlation of a signal with a delayed copy of itself 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _acf = st.acf(_x, adjusted=True, fft=_N > 1250, nlags=int(len(_x) / 2),
         missing='raise')[1:]
        _is_acf_na = np.all(np.isnan(_acf))
        _amax_acf = 0 if _is_acf_na else np.argmax(_acf)
        _amin_acf = 0 if _is_acf_na else np.argmin(_acf)
        _max_acf = 0 if _is_acf_na else np.max(_acf)
        _min_acf = 0 if _is_acf_na else np.min(_acf)

    # Linear Trend
    _lin = sp.linregress(x=np.arange(_N), y=_x)
    _lin_slope, _lin_itct = _lin.slope, _lin.intercept

    # Time-series Complexity
    _std = np.sqrt(_var)
    _norm_x = (_x - _mean) / _std if _std != 0 else np.zeros(len(_x))
    _cid_ce = np.sqrt(np.sum(np.power(np.diff(_norm_x), 2)))

    return dict(
        MED=_med,
        MIN=_min,
        MAX=_max,
        BEP=_bin_entr,
        AVG=_mean,
        VAR=_var,
        SKW=_skew,
        KUR=_kurt,
        ASC=_asc,
        MAXLAG=_amax_acf,
        MAXLAGVAL=_max_acf,
        MINLAG=_amin_acf,
        MINLAGVAL=_min_acf,
        LTS=_lin_slope,
        LTI=_lin_itct,
        CID=_cid_ce,
        # =_sam_enp
    )


def _extract_nominal_feature(_x: np.ndarray, _is_bounded: bool, _pid=None) -> Dict[str, any]:
    _N = len(_x)

    # Support
    try:
        _val, _supp = np.unique(_x, return_counts=True)
    except BaseException as e: # base exception contains inside KeyboardInterrupt error
        print('error message',repr(e))
        print("Error in _extract_nominal_feature")
        print("pid: ", _pid)
        print(_x.shape)
        print(_x)
        exit()
    # Entropy
    _entr = sp.entropy(_supp)

    # Abs. Sum of Changes
    _asc = np.sum(_x[1:] != _x[:-1])

  
    _ret = dict(
        ETP=_entr,
        ASC=_asc,
    )

    if _is_bounded:
        _val_sup = {'SUP:{}'.format(_k): _v for _k, _v in zip(_val, _supp)}
        #_val_sup = {'SUP:{}'.format(_k): _v / _N for _k, _v in zip(_val, _supp)}
        _ret = dict(
            **_ret,
            **_val_sup
        )

    return _ret


# when extracting supoprt features some values of sensor will
# not occur in the data,
# hence will be instantiated as NaN when making dataframe
def impute_support_features(df):
    support_features = df.columns[df.columns.str.contains('#SUP')]
    df[support_features] = df[support_features].fillna(0)
    return df

def parallellize_extract_sub(
           
            labels: pd.DataFrame,
            w_size_in_min: int,
            num_sub: int
            ,selected_features: str = None
            ,resample = True

    ):
    pb = ProgressBar(labels.index.get_level_values('pcode').nunique())
    actor = pb.actor

    func = ray.remote(extract_sub).remote 
    results = []

    for pid in labels.index.get_level_values('pcode').unique():
        participant_label = labels.loc[pid]        
        results.append(func(
            pid, participant_label, w_size_in_min, num_sub, actor
            ,selected_features=selected_features
            , resample = resample
        ))  
    pb.print_until_done()
    results = ray.get(results)
    return impute_support_features(
        pd.concat(results)
    )

def get_sub_window_size( w_size,NUM_SUBWINDOWS=6):
    assert w_size%NUM_SUBWINDOWS==0,\
    f'{w_size} MIN is not divisable by {NUM_SUBWINDOWS}'
    
    sw_size = w_size//NUM_SUBWINDOWS    
    return sw_size

def extract_sub(
  
    _pid: str, _label: pd.DataFrame, w_size_in_min, num_sub
    ,  pba=None, selected_features=None
    , resample = True
):
    _features = []
    _sw_size_in_min = get_sub_window_size(
        w_size_in_min, NUM_SUBWINDOWS=num_sub
    ) 
    _raw =dict()
    #resample or not
    if resample:
        _raw = load(f'./proc/resampled_data-{_pid}.pkl')
    else:
        __raw = load(f'./proc/clean_data-{_pid}.pkl')
        for k, v in __raw.items(): #if no resampling, we still need to remove duplicate values
            v = v.loc[lambda x: ~x.index.duplicated(keep=False)]
            _raw[k] =v.dropna()
    #_raw = preprocess(_pid=_pid, _until=_label.index.max(), resample =resample)

    #for each ema extract |_w_size//_sw_size}| window features
    for ema_time in _label.index: 
        subwindow_start = ema_time-dt.timedelta(minutes=w_size_in_min)
        sub_windows = np.arange(
            subwindow_start
            , ema_time+dt.timedelta(minutes=1) #add 1 minute because np.arrange is [,)
            , dt.timedelta(minutes=_sw_size_in_min)
        )   
        for _s, _e in zip(sub_windows[:-1], sub_windows[1:]):
            _row = []
            # Windowed Features
            for _d_name in _raw.keys():
                if 'Today' in _d_name: 
                    continue
                
                #window for the data source
                _d_value = _raw[_d_name][_s:_e]                    
                if len(_d_value)<1:
                    continue # zero sized window

                _d_win_a = np.asarray(_d_value)    # throws away index(datetime)
                _f = _extract_nominal_feature(
                    _d_win_a
                    , False if _d_name in [
                        'LOC_CLS', 'APP_RAW'
                        
                    ] else True         
                    , _pid=_pid              
                ) if (_d_value.dtype != float) else _extract_numeric_feature(_d_win_a)

                if selected_features is None:
                    _f_win = {f'{_d_name}#{k}': v for k, v in _f.items()} 
                else:    
                    _f_win = {f'{_d_name}#{k}':v for k, v in _f.items() \
                        if f'{_d_name}#{k}' in selected_features
                    }
                                    
                _row.append(_f_win)
            
    
            if len(_row)>0:
                _feature = reduce(lambda a, b: dict(a, **b), _row) # [{'ACT_CUR':STILL}, {'BAT_LEV_CUR':44}]=> {'ACT_CUR':STILL, 'BAT_LEV_CUR':44}
            else:
                _feature = {}
            _feature.update({
                'pid':_pid
                ,'timestamp':ema_time
                ,'sub_timestamp':_e
            })
            _features.append(_feature)# appending feature for the given ESM
        
    _X    = pd.DataFrame(_features) 
    _X = _X.set_index(['pid','timestamp','sub_timestamp']).sort_index()
    log('Complete feature extraction (n = {}, dim = {}).'.format(_X.shape[0], _X.shape[1]))
    if pba is not None:
        pba.update.remote(1)
    return _X
    

@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter



class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return

def parallellize_extract_sliding(
        labels: pd.DataFrame, _sw_size_in_min ,selected_features: list
        , resample = True
    ):

    results = []
    pb = ProgressBar(labels.index.get_level_values('pcode').nunique())
    actor = pb.actor

    func = ray.remote(extract_slidingFeatures).remote 
    for pid in labels.index.get_level_values('pcode').unique():        
        participant_label = labels.loc[pid]        
        results.append(func(
            pid, participant_label, _sw_size_in_min, selected_features
            ,actor
            , resample = resample
        ))                
    pb.print_until_done()
    results = ray.get(results)
    df = impute_support_features(
        pd.concat(results)        
    )
    return df


def extract_slidingFeatures(
        _pid: str, _label: pd.DataFrame, _sw_size_in_min
        , selected_features, pba=None
        , resample = True
    ):
    '''
    - Slides through the whole data (e.g, 30 min window)
    - used for Association erulwe mining
    '''
    assert selected_features !=None, 'please pass selected_features'
    
    _raw =dict()
    #resample or not
    if resample:
        _raw = load(f'./proc/resampled_data-{_pid}.pkl')
    else:
        __raw = load(f'./proc/clean_data-{_pid}.pkl')
        for k, v in __raw.items(): #if no resampling, we still need to remove duplicate values
            v = v.loc[lambda x: ~x.index.duplicated(keep=False)]
            _raw[k] =v.dropna()
    #10 am of the day when participant started collecting
    _start_of_week = _label.index.min().replace(hour=10, minute=0, second=0) 
    
    _features = []
    for day in range(COLLECTION_DAYS):
        start_of_day = _start_of_week+dt.timedelta(days=day) # DAY1 10 AM, DAY2 10 AM, ..
        for minutes_passed in range(
            _sw_size_in_min
            , COLLECTION_HOURS*MIN_IN_HOUR+1
            , _sw_size_in_min
        ):
            _t = start_of_day+dt.timedelta(minutes=minutes_passed) 
                   
            _row = []        
            # loop throuugh different data sources
            for d_index, (_d_name, _d_value) in enumerate(_raw.items()):
                if 'Today' in _d_name:
                    continue # no need to extract time features for (accumulated) daily sensor values             
                
                try:                    
                    _window_start = _t - dt.timedelta(minutes=_sw_size_in_min)                    
                    _d_win = _d_value[_window_start:_t]                   
                    if len(_d_win)<1:
                        log(
                            f'extract_slidingSubFeatures: zero sized window\
                                between{_window_start}-{_t} '
                        )
                        continue

                    _d_win_a = np.asarray(_d_win)    # throws away index(datetime)
                    if _d_value.dtype !=float:
                        _f = _extract_nominal_feature(
                            _d_win_a
                            , False if _d_name in ['LOC_CLS', 'APP_RAW'] else True
                        ) 
                    
                    else:
                        _f = _extract_numeric_feature(_d_win_a)
                    
                    _f_win = {}
                    for k, v in _f.items():
                        feature_name = f'{_d_name}#{k}'
                        if feature_name in selected_features:
                            _f_win[feature_name]=v
                        
                    _row.append(_f_win)                                
                except:
                    log(
                        'extract_slidingSubFeatures'
                        , f'Error at {_t}'
                        , traceback.format_exc()
                    )
            
            if len(_row)>0:
                #change the list of dict to single dictionary. i.e., [{'ACT_CUR':STILL}, {'BAT_LEV_CUR':44}]=> {'ACT_CUR':STILL, 'BAT_LEV_CUR':44}
                _feature = reduce(lambda a, b: dict(a, **b), _row)                
            else:
                _feature = {}      
            _feature.update({
                'pid':_pid
                ,'timestamp':_t
            })
            _features.append(_feature)# appending feature for the given subwindow
    _X    = pd.DataFrame(_features).set_index(['pid','timestamp'])
    log('extract_slidingSubFeatures Complete feature extraction (n = {}, dim = {}).'.format(_X.shape[0], _X.shape[1]))
    pba.update.remote(1) if pba is not None else None
    _X = impute_support_features(_X)
    return _X


def extract(
        _pid: str
        , _labels: pd.DataFrame
        , _w_size_in_min
        , selected_features 
        , pba=None
    ):
    log('Start extracting {}.'.format(_pid))
    _features = []
    _label = _labels.loc[_pid]
    _raw = load(f'./proc/resampled_data-{_pid}.pkl')
    for _t in _label.index:
        _row = {}
        for _d_name  in _raw.keys():
            _s = _t - dt.timedelta(minutes=_w_size_in_min)
            
            _d_win_a = np.asarray( 
                _raw[_d_name][_s:_t]    
            )
            if len(_d_win_a)<1:
                continue
            if _d_win_a.dtype != float:
                is_bounded = False if _d_name in ['LOC_CLS', 'APP_RAW'] else True
                _f = _extract_nominal_feature(
                    _d_win_a
                    , is_bounded
                )
            else:
                _f = _extract_numeric_feature(_d_win_a)
            for _k, _v in _f.items():
                feature_name = '{}#{}'.format(_d_name, _k)

                if selected_features == False:
                     _row.update({feature_name: _v})
                     
                else:
                    if feature_name in selected_features:
                        _row.update({feature_name: _v})
                #else:
                    #log('extract: {} is not in selected features'.format(feature_name))
           
        if len(_row)==0:  
            #log(
            #    f'extract: pid={_pid} ema at {_t} does not have any sensor data'
            #)
            continue
        #_feature = reduce(lambda a, b: dict(a, **b), _row) # [{'ACT_CUR':STILL}, {'BAT_LEV_CUR':44}]=> {'ACT_CUR':STILL, 'BAT_LEV_CUR':44}
        _row.update({'pid':_pid,'timestamp':_t})    
        _features.append(_row)# appending feature for the given intervention
        
    _X    = pd.DataFrame(_features) 
    _X = _X.set_index(['pid','timestamp']).sort_index()
    _X = impute_support_features(_X)
    log('extract {}: Complete feature extraction (n = {}, dim = {}).'.format(_pid, _X.shape[0], _X.shape[1]))
    if pba:
        pba.update.remote(1)
    return _X



def parallellize_extract(
            labels: pd.DataFrame,
            w_size_in_min: int
            , selected_features
            , use_ray: bool = True
    ):
    results = []
    func = ray.remote(extract).remote if use_ray else extract
    pb = ProgressBar(labels.index.get_level_values('pcode').nunique())
    
    for pid in tqdm(labels.index.get_level_values('pcode').unique()):
        results.append(
            func(
                pid
                , labels
                , w_size_in_min
                , selected_features 
                , pba =pb.actor
            )
        )        
        
    pb.print_until_done()
    df = pd.concat(ray.get(results)) if use_ray else pd.concat(results)    
    return df

def _extract_time_feature(_timestamp: pd.Timestamp) -> Dict[str, any]:
    _day_of_week = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][_timestamp.isoweekday() - 1]
    _is_weekend = 1 if _timestamp.isoweekday() > 5 else 0
    _hour = _timestamp.hour

    if 0 <= _hour < 12:
        _hour_name = 'MORNING'
    elif 12 <= _hour < 15:
        _hour_name = 'AFTERNOON'
    elif 15 <= _hour < 18:
        _hour_name = 'LATE_AFTERNOON'
    elif 18 <= _hour < 21:
        _hour_name = 'EVENING'
    elif 21 <= _hour < 24:
        _hour_name = 'NIGHT'
    else:
        _hour_name = 'MIDNIGHT'

    return dict(
        DOW=_day_of_week,
        WKD=_is_weekend,
        HRN=_hour_name
    )
    
def extract_extendedFeatures(_pid: str, _label: pd.DataFrame, pba=None):
    ''''
        features from certain sensors could be null even after _resample. 
        This is due to other sources are not null and pd.concat consideres
        non-added sources as NULL
    '''
    _raw = load(f'./proc/resampled_data-{_pid}.pkl')
    _features = []
    for _t in _label.index: # loop over each timestamp of ESM
        _row = []         
        # Current value of sensor/entity
        for _d_name, _d_value in _raw.items():            
            if _d_value.index.min()>=_t or len(_d_value)<1: # some data such as ringer mode may not exist for some participants or disappear after resampling
                continue # there is no sensor data before _t
                
            try:
                _v = _d_value[_d_value[:_t].index.max()] # get the only last value before intervention at time `t`

                if _d_name not in ['LOC_CLS', 'APP_RAW']:
                    _f_cur = {'{}#CUR#VAL'.format(_d_name): _v}
                    _row.append(_f_cur)
            except:
                log(
                    'extract_extendedFeatures: Error occurs on pid = {}, data = {}, window = CUR at time = {}\n Traceback:\n{}'.format(
                        _d_name, _pid, _t, traceback.format_exc())
                )
        

        # Time-related Feature
        try:
            _f_tim = {
                'TIM#CUR#{}'.format(k): v for k, v in _extract_time_feature(_t).items()
            }
            _row.append(_f_tim)
        except:
            log(
                'extract_extendedFeatures: Error occurs on pid = {}, data = TIM, window = CUR at time = {}\nTraceback:\n{}'.format(
                    _pid, _t, traceback.format_exc())
            )

        # ESM Response
        
        try:
            _f_esm = {
                'ESM#CUR#VLC': _label['val_dyn'][_t],
                'ESM#CUR#ARL': _label['aro_dyn'][_t],
                'ESM#CUR#ATN': _label['att_dyn'][_t],
                #'ESM#CUR#STR': _label['stress'][_t],
                'ESM#CUR#DRN': _label['dst_dyn'][_t],
                #'ESM#CUR#CNG': _label['change'][_t]
            }
            _row.append(_f_esm)
        except:
            log(
                'extract_extendedFeatures: Error occurs on pid = {}, data = ESM, window = CUR at time = {}\nTraceback:\n{}'.format(
                    _pid, _t, traceback.format_exc())
            )

        # Extract Label Trend Feature   (i.e., history of sts_dyn)
        try:
            mean_sts_dyn= _label['sts_dyn'][:_t-timedelta(seconds=1)].mean()
            _row.append({'RCT#AVG#sts_dyn':mean_sts_dyn})
        except:        
            log(
                'extract_extendedFeatures: Error occurs on pid = {}, data = RCT, at time = {}\nTraceback:\n{}'.format(
                    _pid, _t, traceback.format_exc())
            )

        # Extract Last Label Value    
        try:
            last_sts_dyn= _label.loc[_label[:_t-timedelta(seconds=1)].index.max()]['sts_dyn']
            _row.append({'RCT#LAST#sts_dyn':last_sts_dyn})
        except:        
            log(
                'extract_extendedFeatures: Error occurs on pid = {}, data = RCT, at time = {}\nTraceback:\n{}'.format(
                    _pid, _t, traceback.format_exc())
            )

        if len(_row)>0:
            _feature = reduce(lambda a, b: dict(a, **b), _row) # [{'ACT_CUR':STILL}, {'BAT_LEV_CUR':44}] becomes {'ACT_CUR':STILL, 'BAT_LEV_CUR':44}
        else:
            _feature = {}
        _feature.update({'pid':_pid,'timestamp':_t})
        # _feature['pid'] = _pid
        # _feature['timestamp'] = _t
        _features.append(_feature)# appending feature for the given intervention
        
    _X = pd.DataFrame(_features) .set_index(['pid','timestamp']).sort_index()
    log('extract_extendedFeatures: Complete feature extraction (n = {}, dim = {}).'.format(_X.shape[0], _X.shape[1])
    )
    if pba:
        pba.update.remote(1)
    return _X

def extract_extended_parallel(
    labels: pd.DataFrame,
        
):
    results = []
    
    pb = ProgressBar(labels.index.get_level_values('pcode').nunique())
    actor = pb.actor

    func = ray.remote(extract_extendedFeatures).remote
    for pid in (labels.index.get_level_values('pcode').unique()):
        print('Processing {}'.format(pid))
        participant_label = labels.loc[pid]        
        results.append(func(pid, participant_label, actor))        
        
    pb.print_until_done()
    results = ray.get(results)
    df = pd.concat(results)    
    return df
