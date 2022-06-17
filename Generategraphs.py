import numpy as nump
from scipy import fftpack, arange
import librosa
import librosa.display
import matplotlib.pyplot as plots
from matplotlib.pyplot import specgram
class Graphs:
	def waves(self,s_n,rs):
		i = 1
		fg = plots.figure(figsize=(30,60), dpi = 900)
		for ni,fi in zip(s_n,rs):
			plots.subplot(10,1,i)
			librosa.display.waveshow(nump.array(fi),sr=22050)
			plots.title(ni.title())
			i += 1
		plots.suptitle('Waveplot',x=0.5, y=0.915,fontsize=18)
		plots.xlabel('Time(samples)')
		plots.ylabel('Amplitude')
		plots.show()
    
	def spectrogram(self,s_n,rs):
		i = 1
		fg = plots.figure(figsize=(30,60), dpi = 900)
		for ni,fi in zip(s_n,rs):
			plots.subplot(10,1,i)
			specgram(nump.array(fi), Fs=22050)
			plots.title(ni.title())
			i += 1
		plots.suptitle('Spectrogram',x=0.5, y=0.915,fontsize=18)
		plots.xlabel('Time(samples)')
		plots.ylabel('Frequency')
		plots.show()

	def log_Power_Spectrogram(self,s_n,rs):
		i = 1
		fg = plots.figure(figsize=(30,60), dpi = 900)
		for ni,fi in zip(s_n,rs):
			plots.subplot(10,1,i)
			A = librosa.core.amplitude_to_db(nump.abs(librosa.stft(fi))**2, ref=nump.max)
			librosa.display.specshow(A,x_axis='time' ,y_axis='log')
			plots.title(ni.title())
			i += 1
		plots.suptitle('Log power spectrogram',x=0.5, y=0.915,fontsize=18)
		plots.show()
		
	def spectrum(self,a, f):
		a = a - nump.average(a) 
		l = len(a)
		k = arange(l)
		time_array = l / float(f)
		frequency_array = k / float(time_array) 
		frequency_array = frequency_array[range(l // 2)] 
		a = fftpack.fft(a) / l  
		a = a[range(l // 2)]
		plots.figure(figsize=(20,10), dpi = 200)
		plots.subplot(2, 1, 2)
		plots.plot(frequency_array, abs(a), 'b')
		plots.xlabel('Freq (Hz)')
		plots.ylabel('|X(freq)|')
		plots.tight_layout()