import numpy as np

def add_comfort_penalty(df, comfort_temp=22.0, penalty_weight=15):
   #adds temp_deviation and penalized_energy to dataframe
    df = df.copy()
    df['temp_deviation'] = np.abs(df['air_temperature'] - comfort_temp)
    df['penalized_energy'] = df['meter_reading'] + penalty_weight * df['temp_deviation']
    return df


def prepare_training_data(df, comfort_temp=22.0, penalty_weight=15.0):
   
    df = df.copy()
    
    # Add log of meter_reading (to reduce skewness of high results)
    df['log_meter_reading'] = np.log1p(df['meter_reading'])
    
    # Add temperature deviation
    df['temp_deviation'] = (df['air_temperature'] - comfort_temp).abs()
    
    # Calculate penalized energy
    df['penalized_energy'] = df['log_meter_reading'] + penalty_weight * df['temp_deviation']
    
    # Define features and target
    X = df[['air_temperature']]
    y = df['penalized_energy']
    
    return X, y

