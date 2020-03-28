import matplotlib.pyplot as plt
import numpy as np
from load_MITBIH import *
import settings

from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show, legend, xticks

def fir(x):
    #------------------------------------------------
    # Create a signal for demonstration.
    #------------------------------------------------
    sample_rate = 360

    #------------------------------------------------
    # Create a FIR filter and apply it to x.
    #------------------------------------------------

    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = 35

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

    # Use lfilter to filter x with the FIR filter.
    filtered_x = lfilter(taps, 1.0, x)

    #------------------------------------------------
    # Plot the FIR filter coefficients.
    #------------------------------------------------

    # figure(1)
    # plot(taps, 'bo-', linewidth=2)
    # title('Filter Coefficients (%d taps)' % N)
    # grid(True)

    #------------------------------------------------
    # Plot the magnitude response of the filter.
    #------------------------------------------------

    # figure(2)
    # clf()
    # w, h = freqz(taps, worN=8000)
    # plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
    # xlabel('Frequency (Hz)')
    # ylabel('Gain')
    # title('Frequency Response')
    # ylim(-0.05, 1.05)
    # grid(True)

    # # Upper inset plot.
    # plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
    # # xlim(0,8.0)
    # # ylim(0.9985, 1.001)
    # # grid(True)

    # # Lower inset plot
    # plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
    # # xlim(12.0, 20.0)
    # # ylim(0.0, 0.0025)
    # # grid(True)

    #------------------------------------------------
    # Plot the original and filtered signals.
    #------------------------------------------------

    # The phase delay of the filtered signal.
    delay = 132# 0.5 * (N-1) / sample_rate
    t = np.arange(0, len(x))

    # figure(3)
    # Plot the original signal.
    plot(t, x, label='Median filtered signal')
    # Plot the filtered signal, shifted to compensate for the phase delay.
    # plot(t-delay, filtered_x, 'r-')
    # Plot just the "good" part of the filtered signal.  The first N-1
    # samples are "corrupted" by the initial conditions.
    plot(t[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=4, label='Median + Low-Pass filtered signal')
    xlabel('Time Domain')
    ylabel('Frequency Domain')
    legend(loc='upper right', ncol=1, shadow=False, fancybox=True)
    xticks([])

    # grid(True)

    show()














# Generate graphics for paper

db_path = settings.db_path
winL = 90
winR = 90
do_preprocess = False
maxRR = True

use_RR = False
norm_RR = False

reduced_DS = False
leads_flag = [1, 0]
# Load train data 
compute_morph = {'raw'}

label_name = ['Normal (N)', 'Supraventricular Ectopic Beat (S)', 'Ventricular Ectopic Beat (V)', 'Fusion (F)']
line_styles =  ['-', '--', ':', '-.']
line_colors =  ['b', 'r', 'g', 'c']
l_width = 2

plt.figure(figsize=(8, 8))

print("1 Raw")
compute_morph = {'raw'}

[features_1, labels_1, patient_num_beats_1] = load_mit_db('DS1', winL, winR, do_preprocess,
    maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag)

[features_2, labels_2, patient_num_beats_2] = load_mit_db('DS2', winL, winR, do_preprocess,
    maxRR, use_RR, norm_RR, compute_morph, db_path, reduced_DS, leads_flag)

features = np.vstack((features_1, features_2))
labels = np.concatenate((labels_1, labels_2))

raw_avg = np.zeros((4, features.shape[1]))

ftrs = features_1[0]
for i in range(1,5):
    ftrs = np.hstack((ftrs, features_1[i]))
fir(ftrs)
quit()
plt.plot(ftrs, label=label_name[0], linestyle=line_styles[0], linewidth=l_width)
plt.xlabel('Time Domain')
plt.ylabel('Frequency Domain')
plt.xticks([])
plt.show()
# raw_avg[0] = np.average(features[labels == 0], axis=0)
# plt.plot(raw_avg[0], label=label_name[0], linestyle=line_styles[0], linewidth=l_width)
# plt.show()
quit()

for n in range(0,4):
    raw_avg[n] = np.average(features[labels == n], axis=0)
    # plt.plot(raw_avg[n], label=label_name[n], linestyle=line_styles[n], linewidth=l_width)
    ax = plt.subplot(2,2,n+1)
    ax.plot(raw_avg[n], line_colors[n], linewidth=l_width)
    ax.set_title(label_name[n])

leg = plt.legend(loc='best', ncol=1, shadow=False, fancybox=True)
leg.get_frame().set_alpha(0.5)





plt.tight_layout()
plt.show()

# plt.savefig('graphic.jpg', figsize=(8, 8))
