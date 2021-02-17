from tensorflow.keras import layers
from tensorflow.keras import models

def CNN_reg(feature_dim, output_dim=3):
    signal_input = layers.Input(shape=feature_dim, name='Input')
    output = signal_input

    # Feature extraction
    for i in range(3):
        output = layers.Conv1D(256, 3, activation='relu')(output)
        output = layers.BatchNormalization(momentum=0.01)(output)
        output = layers.MaxPool1D(2)(output)
        output = layers.Dropout(0.25)(output)
    output = layers.Conv1D(256, 3, activation='relu')(output)
    output = layers.BatchNormalization(momentum=0.01)(output)

    # Regression
    output = layers.Flatten()(output)                
    output = layers.Dense(256, activation='relu')(output)
    output = layers.Dense(64, activation='relu')(output)
    output = layers.Dense(output_dim, activation='linear')(output)
    
    model = models.Model(signal_input, outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
            
    return model

if __name__=='__main__':
    model = CNN_reg((100,70))
    model.summary()