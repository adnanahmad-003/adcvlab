import numpy as np
import matplotlib.pyplot as plt
from .util import *
from scipy import signal

def round_to_nearest_multiple(number):
        length = len(str(number))
        base = 10 ** (length - 1)
        return base * round(number / base)

# def PPM(inputs):
#     [fm,Am,message_type,ql,ppm_ratio,nb] = inputs

#     fm = round_to_nearest_multiple(fm)

#     duration = 1
#     sampling_rate = 1000
#     x = np.linspace(-500, 500, 1000)

#     if message_type == "sin":
#         message = Am*np.sin(2 * np.pi * fm * x)
#     elif message_type == "cos":
#         message = Am*np.cos(2 * np.pi * fm * x)

#     # Range of the amplitude values
#     num_quantization_levels = ql  # Number of quantization levels
#     amplitude_range = (-Am, Am)

#     # Generate a continuous message signal
#     t = np.linspace(0, 1, sampling_rate)
#     message_signal = np.sin(2 * np.pi * fm * x)  # Example message signal

#     # Calculate the step size between quantization levels
#     step_size = (amplitude_range[1] - amplitude_range[0]) / (num_quantization_levels - 1)


#     # Quantize the analog signal
#     quantized_signal = np.round((message - amplitude_range[0]) / step_size) * step_size + amplitude_range[0]
#     quantized_value = ''

#     symbol_duration = 1 / ppm_ratio
#     t_ppm = np.linspace(0, duration, int(sampling_rate * duration))
#     ppm_encoded_signal = np.zeros_like(t_ppm)

#     symbol_index = 0
#     for symbol in quantized_signal:
#         num_samples = int(ppm_ratio * sampling_rate)
#         ppm_encoded_signal[symbol_index:symbol_index + num_samples] = symbol
#         symbol_index += num_samples

    
#     a = plot_graph(x, message,color="red", title="message_signal")
#     b = plot_graph(x, quantized_signal,color="green", title="Quantized wave")
#     c = plot_graph(t_ppm, ppm_encoded_signal, color="blue", title="PPM Encoded Signal")
    
#     return [a,b,c]

def PPM(inputs):

    def generate_ppm(input_signal, ppm_ratio, pulse_frequency, time_duration):
        t = np.linspace(0, time_duration, len(input_signal), endpoint=False)
    
        normalized_signal = (input_signal - np.min(input_signal)) / (np.max(input_signal) - np.min(input_signal))
        pulse_positions = np.floor(ppm_ratio * normalized_signal * len(t)).astype(int)
    
        ppm_waveform = np.zeros(len(t))
        ppm_waveform[pulse_positions] = 1
    
        return t, ppm_waveform

    [fm, Am, message_type, fs,ppm_ratio] = inputs
    fm = round_to_nearest_multiple(fm)
    x = np.linspace(-500, 500, 1000000)

    sampling_rate = 1000000
    duration = 1
    duty_cycle_range = (20, 80)
    position_range = (0.1, 0.9)

    if message_type == "sin":
        message = Am * np.sin(2 * np.pi * fm * x)
    elif message_type == "cos":
        message = Am * np.cos(2 * np.pi * fm * x)
    elif message_type == 'tri':
        message = triangular(fm, Am, x)


    # t = np.linspace(0, duration, int(sampling_rate * duration))
    # normalized_message = (message - message.min()) / (message.max() - message.min())
    # duty_cycle = np.interp(normalized_message, (0, 1), duty_cycle_range) / 100.0
    # pwm_signal = np.where(np.mod(t, 1/fs) < duty_cycle / fs, 1, 0)



    # ppm_signal = np.zeros_like(pwm_signal)
    # pulse_width = 1 / fs
    # ppm_start = 0
    # for i, pulse in enumerate(pwm_signal):
    #     if pulse == 1:
    #         ppm_signal[int(ppm_start * fs):int((ppm_start + pulse_width) * fs)] = 1
    #         ppm_start += pulse_width
    

    # modulated_wave = message * ppm_signal

    pulse = 1+signal.square(2 * np.pi * fs * x)
    time_duration = len(message) / fs

    t, ppm_waveform = generate_ppm(message, ppm_ratio, fs, time_duration)

    

    a = plot_graph(x, message, title="Message", condition="plot", color="red")
    b = plot_graph(t, ppm_waveform, title="PPM Signal", condition="plot", color="green")
    # c = plot_graph(x, pwm_signal, title="Modulated wave", condition="plot", color="blue")

    return [a, b]
    

def PCM(inputs):
    sampling_rate = 1000
    [fm,Am,message_type,ql,nb] = inputs
    fm = round_to_nearest_multiple(fm)

    duration = 1
    #x = np.linspace(0, duration, int(duration * samples_per_sec))
    x = np.linspace(-500, 500, 1000)
    #carrier = 5*np.sin(2 * np.pi * 3000 * x)

    if message_type == "sin":
        message = Am*np.sin(2 * np.pi * fm * x)
    elif message_type == "cos":
        message = Am*np.cos(2 * np.pi * fm * x)

    # Range of the amplitude values
    num_quantization_levels = ql  # Number of quantization levels
    amplitude_range = (-Am, Am)

    # Generate a continuous message signal
    t = np.linspace(0, 1, sampling_rate)
    message_signal = np.sin(2 * np.pi * fm * x)  # Example message signal

    # Calculate the step size between quantization levels
    step_size = (amplitude_range[1] - amplitude_range[0]) / (num_quantization_levels - 1)


    # Quantize the analog signal
    quantized_signal = np.round((message - amplitude_range[0]) / step_size) * step_size + amplitude_range[0]
    quantized_value = ''

    for value in quantized_signal:
        quantized_value += "," + str((value - amplitude_range[0]) / step_size)

    # Encoding: Convert quantized values to binary
    encoded_signal = np.array([format(int((value - amplitude_range[0]) / step_size), '0{0}b'.format(3)) for value in quantized_signal])
    encoded_str = ''.join(encoded_signal)

    def generate_pulse_from_encoded(encoded_str, pulse_width, sampling_rate, start_index, num_bits):
        pulse_signal = np.zeros(int(num_bits * sampling_rate * pulse_width))
        encoded_bit=''
        for i in range(num_bits):
            bit = encoded_str[start_index + i]
            encoded_bit = encoded_bit + bit
            if bit == '1':
                pulse_signal[i * int(sampling_rate * pulse_width):(i + 1) * int(sampling_rate * pulse_width)] = 1
            
        return encoded_bit,pulse_signal

    # def generate_pulse_from_encoded(encoded_str, pulse_width, sampling_rate, start_index):
    #     num_bits = int(len(encoded_str))
    #     pulse_signal = np.zeros(int(num_bits * sampling_rate * pulse_width))
    #     encoded_bit = ''
    #     for i in range(num_bits):
    #         bit = encoded_str[start_index + i]
    #         encoded_bit = encoded_bit + bit
    #         if bit == '1':
    #             pulse_signal[i * int(sampling_rate * pulse_width):(i + 1) * int(sampling_rate * pulse_width)] = 1
    #     return encoded_bit, pulse_signal

    pulse_width = 0.01  # Pulse width in seconds
    start_index = 0  # Start index in the encoded string
    num_bits = nb # Number of bits to plot

    encoded_bit,pulse_signal = generate_pulse_from_encoded(encoded_str, pulse_width, sampling_rate, start_index, num_bits)


    # Decoding: Convert binary values back to quantized levels
    #decoded_signal = np.array([int(code, 2) * step_size for code in encoded_signal])

    a = plot_graph(x, message,color="red", title="message_signal")
    b = plot_graph(t, quantized_signal,color="green", title="Quantized wave")
    c = plot_graph(np.linspace(0, num_bits, len(pulse_signal)), pulse_signal,color="pink", title="pulse")
    d = encoded_bit
    #e = quantized_value

    return [a,b,c,d] 


def PWM(inputs):
     #[Am,Ac,fm,fc,message_type,fs] = inputs
    [fm,Am,message_type,fs] = inputs
    fm = round_to_nearest_multiple(fm)
    #N  = 1000
    x = np.linspace(-500, 500, 1000000)

    fm = round_to_nearest_multiple(fm)

    sampling_rate = 1000000  # Number of samples per second
    #frequency = 10  # Frequency of the PWM signal in Hz
    #duty_cycle = 0.1  # Duty cycle (ratio of on-time to total cycle time)
    duration = 1  # Duration of the signal in seconds
    duty_cycle_range = (20, 80)

 
   

    if message_type == "sin":
        message = Am*np.sin(2 * np.pi * fm * x)
    elif message_type == "cos":
        message = Am*np.cos(2 * np.pi * fm * x)
    elif message_type =='tri':
        message = triangular(fm, Am, x)

    # Generate time values
    t = np.linspace(0, duration, int(sampling_rate * duration))
    normalized_message = (message - message.min()) / (message.max() - message.min())

    # Generate duty cycle based on the normalized message signal
    duty_cycle = np.interp(normalized_message, (0, 1), duty_cycle_range) / 100.0

    # Generate the PWM signal
    pwm_signal = np.where(np.mod(t, 1/fs) < duty_cycle / fs, 1, 0)


    modulated_wave = message * pwm_signal

    
    #carrier = Ac * np.sin(2*np.pi*fc*x)
    #carrier = Ac * np.cos(2 * np.pi * fc * x)
    #k = Ac / Am  # Modulation index
    #modulated_wave = (1 + k * message) * carrier
    #modulated = Ac * (1 + k * message) * np.sin(2*np.pi*fc*x)

    a = plot_graph(x, message, title="Message",condition="plot",color="red")
    b = plot_graph(t, pwm_signal, title="PWM Signal",condition="plot",color="green")
    c = plot_graph(x, modulated_wave, title="Modulated wave",condition="plot",color="blue")
    #d = plot_graph(x, demodulated_wave, title="Demodulated wave",condition="plot",color="blue")

    return [a,b,c]

def PAM(inputs):
    #[Am,Ac,fm,fc,message_type,fs] = inputs
    [fm,Am,message_type,fs] = inputs
    #N  = 1000
    x = np.linspace(-500, 500, 1000000)

    fm = round_to_nearest_multiple(fm)

    pulse_width = 0.01  # Pulse width in seconds
    pulse_period = 0.1  # Pulse period in seconds
    duration = 1.0  # Duration of the signal in seconds
    sampling_rate = 1000000  # Sampling rate in Hz
    modulation_index = 0.8

    # t = np.linspace(-500, 500, 1000000)
    pulse = 0.5 * (1 + np.sign(np.sin(2 * np.pi * fs * x)))

   

    if message_type == "sin":
        message = Am*np.sin(2 * np.pi * fm * x)
    elif message_type == "cos":
        message = Am*np.cos(2 * np.pi * fm * x)
    elif message_signal=='tri':
        message = triangular(fm, Am, x_message)

    modulated_wave = message * pulse * modulation_index

    
    #carrier = Ac * np.sin(2*np.pi*fc*x)
    #carrier = Ac * np.cos(2 * np.pi * fc * x)
    #k = Ac / Am  # Modulation index
    #modulated_wave = (1 + k * message) * carrier
    #modulated = Ac * (1 + k * message) * np.sin(2*np.pi*fc*x)

    a = plot_graph(x, message, title="Message",condition="plot",color="red")
    b = plot_graph(x, pulse, title="Pulse",condition="plot",color="green")
    c = plot_graph(x, modulated_wave, title="Modulated wave",condition="plot",color="blue")
    #d = plot_graph(x, demodulated_wave, title="Demodulated wave",condition="plot",color="blue")

    return [a,b,c]

def QUANTIZATION(inputs):
    #[Am,Ac,fm,fc,message_type,fs] = inputs
    [fm,Am,message_type] = inputs
    #N  = 1000
    x = np.linspace(-500, 500, 1000000)
    fm = round_to_nearest_multiple(fm)

    #num_samples = 1000  # Number of samples in the message signal
    amplitude_range = (-1, 1)  # Range of the amplitude values
    num_quantization_levels = 4  # Number of quantization levels

    # Generate a continuous message signal
    #t = np.linspace(0, 1, num_samples)
    message_signal = np.sin(2 * np.pi * fm * x)  # Example message signal

    # Calculate the step size between quantization levels
    step_size = (amplitude_range[1] - amplitude_range[0]) / (num_quantization_levels - 1)


    if message_type == "sin":
        message = Am*np.sin(2 * np.pi * fm * x)
    elif message_type == "cos":
        message = Am*np.cos(2 * np.pi * fm * x)
    elif message_type== "tri":
        message = triangular(fm, Am, x)    

    # Quantize the message signal
    quantized_wave = np.round((message - amplitude_range[0]) / step_size) * step_size + amplitude_range[0]

    
    #carrier = Ac * np.sin(2*np.pi*fc*x)
    #carrier = Ac * np.cos(2 * np.pi * fc * x)
    #k = Ac / Am  # Modulation index
    #modulated_wave = (1 + k * message) * carrier
    #modulated = Ac * (1 + k * message) * np.sin(2*np.pi*fc*x)

    a = plot_graph(x, message, title="Message",condition="plot",color="red")
    #b = plot_graph(x, pulse, title="Pulse",condition="plot",color="green")
    b = plot_graph(x, quantized_wave, title="Quantized wave",condition="plot",color="blue")
    #d = plot_graph(x, demodulated_wave, title="Demodulated wave",condition="plot",color="blue")

    return [a,b]

# def SAMPLING(inputs):
#     #[Am,Ac,fm,fc,message_type,fs] = inputs
#     [fm,Am,message_type,fs] = inputs
#     #N  = 1000
  
#     fm = round_to_nearest_multiple(fm)
#         #x = np.linspace(0,1,fs)

#     #pulse_width = 0.01  # Pulse width in seconds
#     #pulse_period = 0.1  # Pulse period in seconds
#     #duration = 1.0  # Duration of the signal in seconds
#     x = np.linspace(-1000, 1000, 1000000)
#     y = np.linspace(-1000, 1000, 1000000)

#     sampling_rate = 1000000  # Sampling rate in Hz

#     if(fs <= 500):
#         x = np.linspace(-1000, 1000, 1000000)
#         sampling_rate = 1000000  # Sampling rate in Hz
    
#     else:
#         x = np.linspace(-100000, 100000, 1000000)
#         sampling_rate = 1000000  # Sampling rate in Hz
    

#     # t = np.linspace(-500, 500, 1000000)
#     # pulse = 1 + np.sign(np.sin(2 * np.pi * fs * x))
#     pulse = 1+signal.square(2 * np.pi * fs * x)

   

#     if message_type == "sin":
#         message = Am*np.sin(2 * np.pi * fm * x)
#     elif message_type == "cos":
#         message = Am*np.cos(2 * np.pi * fm * x)

#     modulated_wave = message * pulse 

    
#     #carrier = Ac * np.sin(2*np.pi*fc*x)
#     #carrier = Ac * np.cos(2 * np.pi * fc * x)
#     #k = Ac / Am  # Modulation index
#     #modulated_wave = (1 + k * message) * carrier
#     #modulated = Ac * (1 + k * message) * np.sin(2*np.pi*fc*x)

#     a = plot_graph(y, message, title="Message",condition="plot",color="red")
#     b = plot_graph(x, pulse, title="Pulse",condition="plot",color="green")
#     c = plot_graph(x, modulated_wave, title="Sampled wave",condition="plot",color="blue")
    
#     return [a,b,c]

def SAMPLING(inputs):
    #[Am,Ac,fm,fc,message_type,fs] = inputs
    [fm,Am,message_type,fs] = inputs
    #N  = 1000
    x = np.linspace(-500, 500, 1000000)

    fm = round_to_nearest_multiple(fm)

    pulse_width = 0.01  # Pulse width in seconds
    pulse_period = 0.1  # Pulse period in seconds
    duration = 1.0  # Duration of the signal in seconds
    sampling_rate = 1000000  # Sampling rate in Hz

    # t = np.linspace(-500, 500, 1000000)
    pulse = 0.5 * (1 + np.sign(np.sin(2 * np.pi * fs * x)))

   

    if message_type == "sin":
        message = Am*np.sin(2 * np.pi * fm * x)
    elif message_type == "cos":
        message = Am*np.cos(2 * np.pi * fm * x)
    elif message_signal=='tri':
        message = triangular(fm, Am, x_message)

    modulated_wave = message * pulse

    
    #carrier = Ac * np.sin(2*np.pi*fc*x)
    #carrier = Ac * np.cos(2 * np.pi * fc * x)
    #k = Ac / Am  # Modulation index
    #modulated_wave = (1 + k * message) * carrier
    #modulated = Ac * (1 + k * message) * np.sin(2*np.pi*fc*x)

    a = plot_graph(x, message, title="Message",condition="plot",color="red")
    b = plot_graph(x, pulse, title="Pulse",condition="plot",color="green")
    c = plot_graph(x, modulated_wave, title="Modulated wave",condition="plot",color="blue")
    #d = plot_graph(x, demodulated_wave, title="Demodulated wave",condition="plot",color="blue")

    return [a,b,c]