class WritingCSV:
	def writetocsv(self,data):
		d_data = data
		s=[]
		for j in d_data['filelocation']:
			j=j.split('archive')
			j[0]=r'H:\archive'
			j="".join(j)
			s.append(j)
		d_data['filelocation']=s
		return d_data