import matplotlib.pyplot as plt
import numpy as np
import math
from .util import *



def round_to_nearest_multiple(number):
        length = len(str(number))
        base = 10 ** (length - 1)
        return base * round(number / base)
#function for ploting amplitude modulation graph

def AM_main_graph(inputs):
    graphs = [] # created an expty array graphs
    Am,Ac,fm,fc,message_signal = inputs.values() #transfered input values to these variables
    condition = "line" # scattered plotting(dots)


    #if fm >= 1000:
    fm = round_to_nearest_multiple(fm)
    fc = round_to_nearest_multiple(fc)
   # elif fm >= 100:
        #rounded_num = round_to_nearest_multiple(fm, 100)
    #elif fm >= 10:
        #rounded_num = round_to_nearest_multiple(fm, 10)
    #else:
        #rounded_num = num

        
    # if(fm<50): 
    #fm = int(math.ceil(fm / 10.0)) * 10 # converting frequency valuues to multiples pf 10 when its less than 50 for better graph view
    # if(fc<50):
    #fc = int(math.ceil(fc / 10.0)) * 10
    # if(fm>=50): 
    #     fm = int(math.ceil(fm / 50.0)) * 50 # converting frequency valuues to multiples pf 50 when its greater than 50 for better graph view
    # if(fc>=50):
    #     fc = int(math.ceil(fc / 50.0)) * 50

    x_carrier = create_domain_AM() #calls craet domain function from util with input fc which creates an np linspace
    x_message = create_domain_AM() #calls craet domain function from util with input fm which creates an np linspace
    #x_modulated = x_carrier if(len(x_carrier)<len(x_message)) else x_message
    x_modulated = create_domain_AM() #domain for modulated signal is used based on who has the lesser samples
    carrier = Ac*np.cos(2*np.pi*fc*x_carrier)

   
    if(message_signal=="sin"): # if message signal is sine
        message = Am*np.sin(2*np.pi*fm*x_message) # generate message signal based on amplitude given
        # modulated_wave = (Ac+Ac*message)*np.cos(2*np.pi*fc*x_modulated)
        # demodulated_wave = Ac*message # generate demodulated wave
    elif message_signal=='cos':
        message = Am*np.cos(2*np.pi*fm*x_message)
        #modulated_wave = (Ac+Ac*message)*np.cos(2*np.pi*fc*x_modulated)
        #demodulated_wave = Ac*message
    elif message_signal=='tri':
        message = triangular(fm, Am, x_message)
        #modulated_wave = (Ac+Ac*message)*np.cos(2*np.pi*fc*x_modulated)  
        #demodulated_wave = Ac*message
 

    modulated_wave = (1 + message / Ac) * carrier
    #demodulated_wave = Ac*message
    demodulated_wave = modulated_wave * carrier



    
    #add new modulated equation
    # modulated_wave = carrier+message*np.cos(2*np.pi*fc*x_modulated)
    
        
    a = plot_graph(condition = condition, x = x_message, y = message, title = "Message Signal",color='y') # plot graph using plot graph function in util
    b = plot_graph(condition = condition, x = x_carrier, y = carrier, title = "Carrier Signal",color='g')
    c = plot_graph(condition = condition, x = x_modulated, y = modulated_wave, title = "Modulated wave",color='r')
    d = plot_graph(condition = condition, x = x_message, y = demodulated_wave, title="demodulated wave")

    return [a,b,c,d]


def AM_double_sideband_modulation(inputs):
    
    Am,Ac,fm,fc,message_signal = inputs.values()
    condition = "line"

    x_carrier = create_domain_AM()
    x_message = create_domain_AM()
    x_modulated = create_domain_AM()
    
    fm = round_to_nearest_multiple(fm)
    fc = round_to_nearest_multiple(fc)
    # if(fm<50): 
    #     fm = int(math.ceil(fm / 10.0)) * 10
    # if(fc<50):
    #     fc = int(math.ceil(fc / 10.0)) * 10
    # if(fm>50): 
    #     fm = int(math.ceil(fm / 50.0)) * 50
    # if(fc>50):
    #     fc = int(math.ceil(fc / 50.0)) * 50

    carrier = Ac*np.cos(2*np.pi*fc*x_carrier)
   


    if message_signal=="sin":
        message = Am*np.sin(2*np.pi*fm*x_message)
       # modulated_wave = message*Ac*np.cos(2*np.pi*fc*x_modulated)
        #demodulated_wave = Ac*message
    elif message_signal=='tri':
        message = triangular(fm, Am, x_message)
        #demodulated_wave = triangular(x, 0.01*Am*Ac)
       # modulated_wave = message * carrier    # Am*np.cos(2*np.pi*fm*x_message)*Ac*np.cos(2*np.pi*fc*x_modulated)
    elif message_signal=='cos':
        #demodulated_wave = Ac**2*Am/2*np.cos(2*np.pi*fm*x_message)    
        message = Am*np.cos(2*np.pi*fm*x_message)
       # modulated_wave = Am*np.cos(2*np.pi*fm*x_message)*Ac*np.cos(2*np.pi*fc*x_modulated)


    modulated_wave = carrier * message
    demodulated_wave = modulated_wave * carrier
    

    a = plot_graph(condition = condition, x = x_message, y = message, title = "Message Signal", color = 'y')
    b = plot_graph(condition = condition, x = x_carrier, y = carrier, title = "Carrier Signal", color = 'g')
    c = plot_graph(condition = condition, x = x_modulated, y = modulated_wave, title = "Modulated wave", color ='r')
    d = plot_graph(condition = condition, x = x_message, y = demodulated_wave, title="demodulated wave", color = 'm')

    return [a,b,c,d]



def AM_ssb_modulation(inputs):
    Am,Ac,fm,fc,message_signal = inputs.values()
    condition = "line"
    x_carrier = create_domain_AM()
    x_message = create_domain_AM()
    x_modulated = create_domain_AM() #x_carrier if(len(x_carrier)<len(x_message)) else x_message    
    
    fm = round_to_nearest_multiple(fm)
    fc = round_to_nearest_multiple(fc)
    carrier = Ac*np.cos(2*np.pi*fc*x_carrier)

    # if(fm<50): 
    #     fm = int(math.ceil(fm / 10.0)) * 10
    # if(fc<50):
    #     fc = int(math.ceil(fc / 10.0)) * 10
    # if(fm>50): 
    #     fm = int(math.ceil(fm / 50.0)) * 50
    # if(fc>50):
    #     fc = int(math.ceil(fc / 50.0)) * 50

    if message_signal=="sin":
        #demodulated_wave = (Am*Ac**2*np.sin(2*np.pi*fm*x_message))/4
        message = Am*np.sin(2*np.pi*fm*x_message)
        #modulated_positive = 1/2*Am*Ac*(np.sin(2*(fc-fm)*np.pi*x_modulated))
        #modulated_negative = 1/2*Am*Ac*(np.sin(2*(fc+fm)*np.pi*x_modulated))
    elif message_signal=="cos":
        message = Am*np.cos(2*np.pi*fm*x_message)
        #demodulated_wave = Am*Ac**2*np.cos(2*np.pi*fm*x_message)/4
        #modulated_negative = 1/2*Am*Ac*(np.cos(2*(fc-fm)*np.pi*x_modulated))
        #modulated_positive = 1/2*Am*Ac*(np.cos(2*(fc+fm)*np.pi*x_modulated))
    elif message_signal =="tri":
        message = triangular(fm, Am, x_message)
        #demodulated_wave = triangular(x_message, 0.01*Am*Ac)
        #modulated_positive = message*carrier + triangular(x_message, A)
        #modulated_negative = message*carrier - triangular(x_message, A)

    modulated_positive = message * carrier
    modulated_negative = -message * carrier

    demodulated_wave = (modulated_positive-modulated_negative)*carrier

    #y2 = (Am*np.cos(2*np.pi*fc*x_message))
    
    a = plot_graph(condition = condition, x = x_message, y = message,color='g', title = "Message Signal")
    b = plot_graph(condition = condition, x = x_carrier, y = carrier,color='m', title = "Carrier Signal")
    c = plot_graph(condition = condition, x = x_modulated, y = modulated_positive, color='r', title = "Modulated wave 1",text="upper Sideband")
    d = plot_graph(condition = condition, x = x_modulated, y = modulated_negative, color='b', title = "Modulated wave 2",text="lower Sideband")
    e = plot_graph(condition = condition, x = x_message, y=demodulated_wave,color='r', title="demodulated wave")
    
    return [a,b,c,d,e]

def AM_QAM(inputs):
    Am,Ac,fm,fc,message_signal,message_signal_2 = inputs.values()
    condition="line"
    x_carrier = create_domain_AM()
    x_message = create_domain_AM()
    x_modulated = create_domain_AM() #x_carrier if(len(x_carrier)<len(x_message)) else x_message

    # if(fm<50): 
    #     fm = int(math.ceil(fm / 10.0)) * 10
    # if(fc<50):
    #     fc = int(math.ceil(fc / 10.0)) * 10
    # if(fm>50): 
    #     fm = int(math.ceil(fm / 50.0)) * 50
    # if(fc>50):
    #     fc = int(math.ceil(fc / 50.0)) * 50

    fm = round_to_nearest_multiple(fm)
    fc = round_to_nearest_multiple(fc)

    c1 = Ac*np.cos(2*np.pi*fc*x_carrier) #carrier 1
    c2 = Ac*np.sin(2*np.pi*fc*x_carrier) #carrier 2

    if message_signal=="sin":
        m1 = Am*np.sin(2*np.pi*fm*x_message)
    elif message_signal=="cos":
        m1 = Am*np.cos(2*np.pi*fm*x_message)
    elif message_signal=="tri":
        m1 = triangular(fm, Am, x_message)
    
    if message_signal_2 == "sin":
        m2 = Am*np.sin(2*np.pi*fm*x_message)
    elif message_signal_2 == "cos":
        m2 = Am*np.cos(2*np.pi*fm*x_message)
    elif message_signal_2 == "tri":
        m1 = triangular(x_message, Am)

    # modulated_wave_1 = (c1 + m1) * np.cos(2 * np.pi * fc * x_modulated)    
    # modulated_wave_2 = (c2 + m2) * np.sin(2 * np.pi * fc * x_modulated) 

    modulated_wave_1 = c1 * m1
    modulated_wave_2 = c2 * m2  

    demodulated_wave_1 = modulated_wave_1 * np.cos(2 * np.pi * fc * x_modulated)
    demodulated_wave_2 = modulated_wave_2 * np.sin(2 * np.pi * fc * x_modulated)  
    
    modulated_wave = modulated_wave_1 + modulated_wave_2

    a = plot_graph(condition = condition,x = x_message, y = m1,color='b', title = "Message Signal-1")
    b = plot_graph(condition = condition,x = x_message, y = m2,color='g', title = "Message Signal-2")
    c = plot_graph(condition = condition,x = x_carrier, y = c1,color='m', title = "Carrier Signal-1")
    d = plot_graph(condition = condition,x = x_carrier, y = c2,color='y', title = "Carrier Signal-2")
    e = plot_graph(condition = condition,x = x_modulated, y = modulated_wave_1,color='r', title = "Modulated wave - 1")
    f = plot_graph(condition = condition,x = x_modulated, y = modulated_wave_2,color='r', title = "Modulated wave - 2")
    g = plot_graph(condition = condition,x = x_message, y=demodulated_wave_1,color='r', title="demodulated wave - 1")
    h = plot_graph(condition = condition,x = x_message, y=demodulated_wave_2,color='c', title="demodulated wave - 2")
    
    return [a,b,c,d,e,f,g,h]