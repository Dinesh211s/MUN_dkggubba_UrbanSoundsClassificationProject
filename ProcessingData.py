import numpy as nump
import librosa
import librosa.display
class Preprocessing:
	def job(self,da):
		d_df=da
		filelocation = d_df['filelocation']
		Classes = d_df.Class.unique().tolist()
		indices = list(range(0,len(Classes)))
		class_dict = dict(zip(indices,Classes))
		X = []
		Y = []
		for i in d_df.filelocation:
			new, rate = librosa.load(i)
			mfccs = nump.mean(librosa.feature.mfcc(y=new, sr=rate, n_mfcc=40).T, axis = 0)
			idx = Classes.index(d_df[d_df.filelocation == i]['Class'].tolist()[0])
			Y.append(idx)
			X.append(mfccs)
		return X, Y, class_dict
		
	def librosa_load_Files(self,f_p_s):
		ds = []
		for fp in f_p_s:
			data,sr = librosa.load(fp)
			ds.append(data)
		return ds
	
	