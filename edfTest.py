from pyedflib import highlevel  # to install this package using pip: 'pip install pyEDFlib'
                                # to install this package using conda: 'conda install -c conda-forge pyedflib'

signals, signal_headers, header = highlevel.read_edf('test.edf')  # reads in the signal data, header for each signal, and the overall edf header from test.edf

C3_data = signals[2]  # in this edf the 3rd signal is data from the C3 electrode (look at signal_headers to determine the label for each signal in your edf)
C3_fs = signal_headers[2]['sample_rate']  # Extract the sampling frequency for the C3 signal

# C3_data will be the 'data' argument to multitaper_spectrogram
# C3_fs will be the 'fs' argument to multitaper_spectrogram
