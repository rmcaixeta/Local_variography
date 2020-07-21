
import numpy as np
import pandas as pd


def rot_mat(azimuth,dipval,rakval):
	rot_mat_ = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
	rot_mat_[0,0] = np.cos(np.pi/2. - np.deg2rad(azimuth))*np.cos(np.deg2rad(-1.0*dipval))
	rot_mat_[0,1] = (np.sin(np.pi/2. - np.deg2rad(azimuth))*np.cos(np.deg2rad(-1.0*dipval)))
	rot_mat_[0,2] = -np.sin(np.deg2rad(-1.0*dipval))
	rot_mat_[1,0] = (np.cos(np.pi/2. - np.deg2rad(azimuth))*np.sin(np.deg2rad(-1.0*dipval))*np.sin(np.deg2rad(rakval))-np.sin(np.pi/2. - np.deg2rad(azimuth))*np.cos(np.deg2rad(rakval)))
	rot_mat_[1,1] = (np.sin(np.pi/2. - np.deg2rad(azimuth))*np.sin(np.deg2rad(-1.0*dipval))*np.sin(np.deg2rad(rakval))+np.cos(np.pi/2. - np.deg2rad(azimuth))*np.cos(np.deg2rad(rakval)))
	rot_mat_[1,2] = (np.cos(np.deg2rad(-1.0*dipval))*np.sin(np.deg2rad(rakval)))
	rot_mat_[2,0] = (np.cos(np.pi/2. - np.deg2rad(azimuth))*np.sin(np.deg2rad(-1.0*dipval))*np.cos(np.deg2rad(rakval))+np.sin(np.pi/2. - np.deg2rad(azimuth))*np.sin(np.deg2rad(rakval)))
	rot_mat_[2,1] = (np.sin(np.pi/2. - np.deg2rad(azimuth))*np.sin(np.deg2rad(-1.0*dipval))*np.cos(np.deg2rad(rakval))-np.cos(np.pi/2. - np.deg2rad(azimuth))*np.sin(np.deg2rad(rakval)))
	rot_mat_[2,2] = (np.cos(np.deg2rad(-1.0*dipval))*np.cos(np.deg2rad(rakval)))
	return rot_mat_


def exp_vario_pairs(df_chunk,df_full,id_to_print,pars,hlimit_d,vlimit_d,coordx,coordy,coordz,azm_code,dip_code,rak_code):
	
	idx_pair={}
	for ax in ['MAX','MED','MIN']: idx_pair[ax]=[]
	
	#ref_vect = {'MAX':np.array([0,1,0]),'MED':np.array([1,0,0]),'MIN':np.array([0,0,1])}

	for index, row in df_chunk.iterrows():
		
		#if index in id_to_print: print(id_to_print[index],"%")
		if index in id_to_print: print(".", end =" ", flush=True)
		
		coordst = np.column_stack(( df_full[coordx].values - row[coordx], df_full[coordy].values - row[coordy], df_full[coordz].values - row[coordz] ))
		
		coordsrot = rot_mat(row[azm_code],row[dip_code],row[rak_code]).dot(coordst.T)
		
		df_full['XTRF__'] = coordsrot[0,:].T
		df_full['YTRF__'] = coordsrot[1,:].T
		df_full['ZTRF__'] = coordsrot[2,:].T
		
		dict_rot = {'MAX':['XTRF__','YTRF__','ZTRF__'],'MED':['YTRF__','XTRF__','ZTRF__'],'MIN':['ZTRF__','XTRF__','YTRF__']}
		
		for ax in ['MAX','MED','MIN']:
			
			df_filt = df_full.copy()
			ax_radius = pars[ax]['LAG'] * pars[ax]['NLAGS'] + pars[ax]['LAG']/2.0

			# Bandwidth and max lag restrictions
			df_filt = df_filt[
				(df_filt['IDX']!=row.IDX) &
				(df_filt[dict_rot[ax][0]].between(-ax_radius,ax_radius)) &
				(df_filt[dict_rot[ax][1]].between(-pars[ax]['BAND'],pars[ax]['BAND'])) &
				(df_filt[dict_rot[ax][2]].between(-pars[ax]['VBAND'],pars[ax]['VBAND']))
				].reset_index(drop=True)
			
			# Horizontal angular tolerance
			if ax!='MIN':
				angs = np.rad2deg(np.arctan2(df_filt[dict_rot[ax][1]],df_filt[dict_rot[ax][0]]))
				
				for i in range(len(angs)):
					while(angs[i]>90): angs[i]-= 180
					while(angs[i]<-90): angs[i]+= 180
				#	if(angs[i]>90): angs[i] = angs[i]-90
				df_filt['ANGH__'] = np.abs(angs)
			
			# Vertical angular tolerance
			unit_vects = [np.array([row_['XTRF__'],row_['YTRF__'],row_['ZTRF__']])/np.linalg.norm(np.array([row_['XTRF__'],row_['YTRF__'],row_['ZTRF__']])) for x,row_ in df_filt.iterrows()]
			angs = [np.rad2deg(np.arccos(np.clip(np.dot(np.array([0,0,1]), x), -1.0, 1.0))) for x in unit_vects]

			if ax=='MIN':
				for i in range(len(angs)):
					while(angs[i]>90): angs[i]-= 180
					while(angs[i]<-90): angs[i]+= 180

				df_filt['ANGV__'] = np.abs(angs)

				df_filt.drop(df_filt[
					((df_filt['ZTRF__'].between(-vlimit_d[ax],vlimit_d[ax])) & (df_filt['ANGV__']>pars[ax]['VANGTOL']))
					].index, inplace=True)
			else:
				for i in range(len(angs)):
					while(angs[i]>=180): angs[i]-= 180
					while(angs[i]<0): angs[i]+= 180
					if(angs[i]>90): angs[i] = angs[i]-90
					else: angs[i] = 90-angs[i]
				df_filt['ANGV__'] = angs

				df_filt.drop(df_filt[
					((df_filt[dict_rot[ax][0]].between(-hlimit_d[ax],hlimit_d[ax])) & (df_filt['ANGH__']>pars[ax]['ANGTOL'])) |
					((df_filt[dict_rot[ax][0]].between(-vlimit_d[ax],vlimit_d[ax])) & (df_filt[dict_rot[ax][1]].between(-vlimit_d[ax],vlimit_d[ax])) & (df_filt['ANGV__']>pars[ax]['VANGTOL']))
					].index, inplace=True)
			
			# SAVE VALID PAIRS
			for ok_idx in df_filt['IDX'].values:
				pair_i = tuple((int(row.IDX),int(ok_idx)))
				if(ok_idx<row.IDX): pair_i = tuple((int(ok_idx),int(row.IDX)))
				idx_pair[ax].append(pair_i)


			# FOR VALIDATE SOME GROUP OF VALID PAIRS IF DESIRED

			#if row.IDX==3016:
			#if row.IDX==444:  cubo
			#	local = df_full.loc[df_full['IDX']==row.IDX].reset_index()
			#	local['ANGH__'] = [9999999 for xx in local.index.values]
			#	local['ANGV__'] = [9999999 for xx in local.index.values]
			#	out = pd.concat([df_filt,local],ignore_index=True,sort=True)
			#	out['TRDIP'] = -1.0*out['TRDIP']
			#	out.to_csv(ax+'_valid.csv',index=False)


	return(idx_pair)


def pairs_df(chunk_pairs,df_data,coordx,coordy,coordz,vvar):
				
	var = df_data[vvar].values
	pts = np.column_stack(( df_data[coordx], df_data[coordy], df_data[coordz] ))

	dist=[]
	head=[]
	tail=[]
	
	for x in chunk_pairs:
		dist.append(np.linalg.norm(pts[x[0],:].flatten()-pts[x[1],:].flatten()))
		head.append(var[x[0]])
		tail.append(var[x[1]])

	df = pd.DataFrame(data=np.column_stack(( chunk_pairs, dist, head, tail )), columns=['P12','D','H','T'])
	df['D'] = pd.to_numeric(df['D'])
	df['H'] = pd.to_numeric(df['H'])
	df['T'] = pd.to_numeric(df['T'])
	return(df)
	



def model_vario(data,modtyp,ngt,cc1,cc2,cc3,show_hist,par):
	
	import matplotlib.pyplot as plt
	
	plt.gcf().clear()

	# Plot par
	total = ngt + cc1 + cc2 + cc3
	
	

	# Spherical three structure vario function
	def cov_value_sph(axis,h):
		if (h == 0.0):
			#cov = ngt + cc1 + cc2
			return ngt
		cov = 0.0
		if (h < par[axis]['a1']): cov = cov + cc1 - cc1*( (1.5*h/par[axis]['a1']) - (h**3./(2.*par[axis]['a1']**3.)) )
		if (h < par[axis]['a2']): cov = cov + cc2 - cc2*( (1.5*h/par[axis]['a2']) - (h**3./(2.*par[axis]['a2']**3.)) )
		if (h < par[axis]['a3']): cov = cov + cc3 - cc3*( (1.5*h/par[axis]['a3']) - (h**3./(2.*par[axis]['a3']**3.)) )
		return total-cov
	
	# Exponential three structure vario function	
	def cov_value_exp(axis,h):
		if (h == 0.0):
			#cov = ngt + cc1 + cc2
			return ngt
		cov = 0.0
		if (cc1>0 and par[axis]['a1']>0): cov += cc1*( np.exp(-3.0*h/par[axis]['a1']) )
		if (cc2>0 and par[axis]['a2']>0): cov += cc2*( np.exp(-3.0*h/par[axis]['a2']) )
		if (cc3>0 and par[axis]['a3']>0): cov += cc3*( np.exp(-3.0*h/par[axis]['a3']) )
		return total-cov

	if (modtyp=="Exponential"): cov_value = cov_value_exp
	else: cov_value = cov_value_sph

	# Reading data
	df=pd.read_csv(data)
	df.sort_values(['AXIS', 'BIN'], inplace=True)

	grouped_df = df.groupby(['AXIS'])
	
	graph_out = 220

	# Plot variograms

	plt.figure(1)
	subp = 1
	for name, group in grouped_df:
		dft = df[df['AXIS']==name]
		
		hist_max = 0.1

		tmp = dft[dft['BIN']!=0].copy()
		tmp['LAG']=tmp['DIST']/tmp['BIN'] 
		lag = tmp['LAG'].mean()

		global_model = grouped_df[['DIST','CORR','NPAIRS']].get_group(name).values	

		fit_model = [cov_value(name,d) for d in np.arange(0,dft['DIST'].max()+1,1)]

		plt.subplot(graph_out+subp)
		
		if (show_hist):
			diff = [(global_model[w+1,0]-global_model[w,0]) for w in range(len(global_model)-1)]
			wd = [np.min(diff)/1.5 for i in range(len(global_model))]

			plt.bar(global_model[:,0],width=wd,height=(global_model[:,2]*hist_max/np.amax(global_model[:,2])),edgecolor='gray',color='orange',alpha=0.5)

		plt.plot(global_model[:,0], 1.0-global_model[:,1], color='orange', linestyle='dashed')
		plt.scatter(global_model[:,0], 1.0-global_model[:,1], s=15, color='black', label="Global model")

		plt.plot(np.arange(0,dft['DIST'].max()+1,1), fit_model, color='red')
		


		plt.title("Axis: " + "{}".format(name))
		plt.ylabel("Semivariogram")
		plt.xlabel("Lag")

		plt.ylim(-0.15,1.2)
		plt.xlim(np.amin(global_model[:,0])-lag,np.amax(global_model[:,0])+lag)

		plt.grid(True)
		
		#plt.savefig("Variograms_"+"{}".format(name)+".png", bbox_inches='tight')
		#plt.gcf().clear()
		
		subp += 1
	
	plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
	fig = plt.gcf()
	figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds

	return([fig,figure_w, figure_h])



def angs_to_data(data1,x1,y1,z1,data2,x2,y2,z2,azm_code,dip_code,rak_code,neg_dip,outname):

	from sklearn.neighbors import NearestNeighbors
	

	# Reading Dataset
	df1=pd.read_csv(data1)
	df2=pd.read_csv(data2)
	
	knn = NearestNeighbors(n_neighbors=1)
	
	knn.fit( np.column_stack((df2[x2],df2[y2],df2[z2])) ) #HD
	idx = knn.kneighbors(np.column_stack((df1[x1],df1[y1],df1[z1])), return_distance=False) #TD
	
	for avar in [azm_code,dip_code,rak_code]:
		if (avar==dip_code and neg_dip==True):  df1[avar] = [-1.*df2.loc[i,avar] for i in idx.flatten()]
		else: df1[avar] = [df2.loc[i,avar] for i in idx.flatten()]
	
	
	df1.to_csv(outname+'.csv',index=False)
