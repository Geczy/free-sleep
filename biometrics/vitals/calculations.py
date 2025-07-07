"""
This module provides functions for processing 8 sleep data and calculating biometrics

The module includes functionalities for:
- Cleaning and preprocessing raw sensor data.
- Running sliding window calculations to estimate heart rate trends.
- Handling and processing RunData objects for comprehensive heart rate interval estimation.

Key Functions:
- `clean_df_pred`: Cleans predicted breathing rate and HRV data, filling missing values and smoothing data.
- `_calculate`: Processes raw sensor data to derive heart rate, HRV, and breathing rate.
- `estimate_heart_rate_intervals`: Runs heart rate estimation over intervals and stores results.
"""
import gc
import numpy as np

from heart.exceptions import BadSignalWarning
from vitals.run_data import RunData
import traceback
import pandas as pd

import sys
import os
import platform
import warnings

if platform.system().lower() == 'linux':
    sys.path.append('/home/dac/free-sleep/biometrics/')

sys.path.append(os.getcwd())
from data_types import *
from vitals.cleaning import interpolate_outliers_in_wave
from heart.filtering import filter_signal, remove_baseline_wander
from heart.preprocessing import scale_data
from heart.heartpy import process
from get_logger import get_logger

logger = get_logger()
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# ---------------------------------------------------------------------------------------------------
# region CLEAN DF

def clean_df_pred(df_pred: pd.DataFrame) -> pd.DataFrame:
    # Breathing Rate Cleaning
    z_score_threshold = 2
    window_size = 30

    # Calculate rolling z-score for breathing_rate
    df_pred['breathing_rate_mean'] = df_pred['breathing_rate'].rolling(window=window_size, min_periods=1).mean()
    df_pred['breathing_rate_std'] = df_pred['breathing_rate'].rolling(window=window_size, min_periods=1).std()
    df_pred['breathing_rate_z_score'] = (df_pred['breathing_rate'] - df_pred['breathing_rate_mean']) / df_pred['breathing_rate_std']

    # Identify and remove outliers
    df_pred.loc[abs(df_pred['breathing_rate_z_score']) > z_score_threshold, 'breathing_rate'] = np.nan

    # Interpolate missing values using polynomial interpolation
    df_pred['breathing_rate'] = df_pred['breathing_rate'].interpolate(method='polynomial', order=2)
    df_pred['breathing_rate'] = df_pred['breathing_rate'].ffill().bfill()

    # Apply a final smoothing
    df_pred['breathing_rate'] = df_pred['breathing_rate'].rolling(window=5, min_periods=1).mean()

    # HRV Cleaning
    # Calculate rolling z-score for hrv
    df_pred['hrv_mean'] = df_pred['hrv'].rolling(window=window_size, min_periods=1).mean()
    df_pred['hrv_std'] = df_pred['hrv'].rolling(window=window_size, min_periods=1).std()
    df_pred['hrv_z_score'] = (df_pred['hrv'] - df_pred['hrv_mean']) / df_pred['hrv_std']

    # Identify and remove outliers
    df_pred.loc[abs(df_pred['hrv_z_score']) > z_score_threshold, 'hrv'] = np.nan

    # Interpolate missing values using polynomial interpolation
    df_pred['hrv'] = df_pred['hrv'].interpolate(method='polynomial', order=2)
    df_pred['hrv'] = df_pred['hrv'].ffill().bfill()

    # Apply a final smoothing
    df_pred['hrv'] = df_pred['hrv'].rolling(window=5, min_periods=1).mean()

    # Drop temporary columns
    df_pred.drop(columns=['breathing_rate_mean', 'breathing_rate_std', 'breathing_rate_z_score', 'hrv_mean', 'hrv_std', 'hrv_z_score'], inplace=True)

    return df_pred


# endregion


# ---------------------------------------------------------------------------------------------------
# region CALCULATIONS


def _calculate(run_data: RunData, side: str):
    # Get the signal
    np_array = np.concatenate(run_data.piezo_df[run_data.start_interval:run_data.end_interval][side])

    # Remove outliers from signal
    data = interpolate_outliers_in_wave(
        np_array,
        lower_percentile=run_data.signal_percentile[0],
        upper_percentile=run_data.signal_percentile[1]
    )

    data = scale_data(data, lower=0, upper=1024)
    data = remove_baseline_wander(data, sample_rate=500.0, cutoff=0.05)

    data = filter_signal(
        data,
        cutoff=[0.5, 20.0],
        sample_rate=500.0,
        order=2,
        filtertype='bandpass'
    )

    working_data, measurement = process(
        data,
        500,
        breathing_method='welch',
        bpmmin=40,
        bpmmax=90,
        windowsize=run_data.window_size,
        clean_rr_method='quotient-filter',
        calculate_breathing=True,
    )
    if run_data.is_valid(measurement):
        return {
            'start_time': run_data.start_interval.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': run_data.end_interval.strftime('%Y-%m-%d %H:%M:%S'),
            'heart_rate': measurement['bpm'],
            'hrv': measurement['sdnn'],
            'breathing_rate': measurement.get('breathingrate', 0),
        }
    return None


# WARNING: ERRORS HERE FAIL SILENTLY - PASS debug=True in order to see errors
def estimate_heart_rate_intervals(run_data: RunData, debug=False):
    """
    Estimates heart rate intervals using the given RunData object.

    Parameters:
    -----------
    run_data : RunData
        The data structure containing sleep data, sensor readings, and runtime parameters.

    Returns:
    --------
    None
        Results are stored in `run_data.df_pred`.

    Example:
    --------
    >>> estimate_heart_rate_intervals(run_data)
    """
    if not debug and run_data.log:
        logger.warning('debug=False, errors will fail SILENTLY, pass debug=True in order to see errors')

    if run_data.log:
        print('-----------------------------------------------------------------------------------------------------')
        print(f'Estimating heart rate for {run_data.name} {run_data.start_time} -> {run_data.end_time}')

    run_data.start_timer()
    while run_data.start_interval <= run_data.end_datetime:
        measurement_1 = None
        measurement_2 = None
        try:
            measurement_1 = _calculate(run_data, run_data.side_1)
        except BadSignalWarning:
            run_data.sensor_1_error_count += 1
        except Exception as e:
            if run_data.log:
                traceback.print_exc()
            run_data.sensor_1_error_count += 1

        if run_data.senor_count == 2:
            try:
                measurement_2 = _calculate(run_data, run_data.side_2)
            except BadSignalWarning:
                run_data.sensor_2_error_count += 1
            except Exception as e:
                if run_data.log:
                    traceback.print_exc()
                run_data.sensor_2_error_count += 1

        if measurement_1 is not None and measurement_2 is not None:
            run_data.measurements_side_1.append(measurement_1)
            run_data.measurements_side_2.append(measurement_2)

            m1_heart_rate = measurement_1['heart_rate']
            m2_heart_rate = measurement_2['heart_rate']
            if run_data.hr_moving_avg is not None:
                heart_rate = (((m1_heart_rate + m2_heart_rate) / 2) + run_data.hr_moving_avg) / 2
            else:
                heart_rate = (m1_heart_rate + m2_heart_rate) / 2

            if run_data.hr_moving_avg is not None and abs(heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
                if heart_rate < run_data.hr_moving_avg:
                    heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
                else:
                    heart_rate = run_data.hr_moving_avg + run_data.hr_std_2

            run_data.heart_rates.append(heart_rate)

            run_data.combined_measurements.append({
                'start_time': run_data.start_interval.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': run_data.end_interval.strftime('%Y-%m-%d %H:%M:%S'),
                'heart_rate': heart_rate,
                'hrv': (measurement_1['hrv'] + measurement_2['hrv']) / 2,
                'breathing_rate': (measurement_1['breathing_rate'] + measurement_2['breathing_rate']) / 2 * 60,
            })

        elif measurement_1 is not None:
            run_data.measurements_side_1.append(measurement_1)
            m1_heart_rate = measurement_1['heart_rate']

            # If the HR differs by more than the allowable movement
            if run_data.hr_moving_avg is not None and abs(m1_heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
                if m1_heart_rate < run_data.hr_moving_avg:
                    m1_heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
                else:
                    m1_heart_rate = run_data.hr_moving_avg + run_data.hr_std_2

            run_data.heart_rates.append(m1_heart_rate)

            measurement_1['heart_rate'] = m1_heart_rate
            run_data.combined_measurements.append(measurement_1)

        elif measurement_2 is not None:
            run_data.sensor_1_drop_count += 1
            m2_heart_rate = measurement_2['heart_rate']

            if run_data.hr_moving_avg is not None:
                heart_rate = (m2_heart_rate + run_data.hr_moving_avg) / 2
            else:
                heart_rate = m2_heart_rate

            if run_data.hr_moving_avg is not None and abs(heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
                if heart_rate < run_data.hr_moving_avg:
                    heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
                else:
                    heart_rate = run_data.hr_moving_avg + run_data.hr_std_2

            run_data.heart_rates.append(heart_rate)

            measurement_2['heart_rate'] = heart_rate
            run_data.combined_measurements.append(measurement_2)
            run_data.measurements_side_2.append(measurement_2)

        run_data.next()

    run_data.stop_timer()
    run_data.print_results()
    run_data.combine_results()
    gc.collect()

# endregion