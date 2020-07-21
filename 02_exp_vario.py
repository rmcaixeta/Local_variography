# -*- coding: utf-8 -*-

import PySimpleGUI as sg
import os
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import lva_vario_funcs as lva


if __name__ == '__main__':
	
	nb_procs = cpu_count()-1
	if (nb_procs<1): nb_procs=1
	
	sg.theme('DarkGrey3')
	sg.SetOptions( button_color=('black','#ff7636') )
	dmdir = 'C:'


	tab_DATA = [
		[sg.Text(' ')],
		[sg.Text('Dataset:', size=(20, 1)), sg.InputText(dmdir+'/dataset_angs.csv',justification='right',key='DATA'),sg.FileBrowse(initial_folder=dmdir,file_types=(("CSV Files", "*.csv"),))],
		[sg.Text('Coordinate columns:', size=(20, 1)),sg.InputText('X',key='X',size=(10, 1)),sg.InputText('Y',key='Y',size=(10, 1)),sg.InputText('Z',key='Z',size=(10, 1))],
		[sg.Text('Local rotation columns:', size=(20, 1)),sg.InputText('TRDIPDIR',key='AZ',size=(10, 1)),sg.InputText('TRDIP',key='DP',size=(10, 1)),sg.InputText('RAKE',key='RK',size=(10, 1))],
		[sg.Text('Variable:', size=(20, 1)), sg.InputText('VAR',key='VVAR')],
		[sg.Text('Output exp. vario:', size=(20, 1)), sg.InputText('exp_variography',key='OUT')],
		[sg.Text(' ')],
		[sg.Checkbox('Use domain:', size=(17,1),default=False,enable_events=True,key='USEDOM'), sg.InputText('BODY',key='DOM',disabled=True)],
		[sg.Text(' ')]
	]

	tab_MAX = [
		[sg.Text(' ')],
		[sg.Text('Lag distance:', size=(20, 1)), sg.InputText('30.0',key='LAG1')],
		[sg.Text('Number of lags:', size=(20, 1)), sg.InputText('5',key='NLAGS1')],
		[sg.Text('Horizontal bandwidth:', size=(20, 1)), sg.InputText('150.0',key='BAND1')],
		[sg.Text('Hor. Angular tolerance:', size=(20, 1)), sg.InputText('90.0',key='ANGTOL1')],
		[sg.Text('Vertical bandwidth:', size=(20, 1)), sg.InputText('5.0',key='VBAND1')],
		[sg.Text('Vert. Angular tolerance:', size=(20, 1)), sg.InputText('45.0',key='VANGTOL1')],
		[sg.Text(' ')]
	]

	tab_MED = [
		[sg.Text(' ')],
		[sg.Text('Lag distance:', size=(20, 1)), sg.InputText('30.0',key='LAG2')],
		[sg.Text('Number of lags:', size=(20, 1)), sg.InputText('5',key='NLAGS2')],
		[sg.Text('Horizontal bandwidth:', size=(20, 1)), sg.InputText('150.0',key='BAND2')],
		[sg.Text('Hor. Angular tolerance:', size=(20, 1)), sg.InputText('90.0',key='ANGTOL2')],
		[sg.Text('Vertical bandwidth:', size=(20, 1)), sg.InputText('5.0',key='VBAND2')],
		[sg.Text('Vert. Angular tolerance:', size=(20, 1)), sg.InputText('45.0',key='VANGTOL2')],
		[sg.Text(' ')]
	]

	tab_MIN = [
		[sg.Text(' ')],
		[sg.Text('Lag distance:', size=(20, 1)), sg.InputText('1.0',key='LAG3')],
		[sg.Text('Number of lags:', size=(20, 1)), sg.InputText('50',key='NLAGS3')],
		[sg.Text('Bandwidth:', size=(20, 1)), sg.InputText('5.0',key='BAND3')],
		[sg.Text('Angular tolerance:', size=(20, 1)), sg.InputText('45.0',key='ANGTOL3')],
		[sg.Text(' ')]
	]

	layout = [
		[sg.TabGroup([[sg.Tab('Data',tab_DATA),sg.Tab('Max',tab_MAX),sg.Tab('Med',tab_MED),sg.Tab('Min',tab_MIN)]])],
		[sg.Button('Run'), sg.Button('Close')]
	]


	window = sg.Window('Variography with local angles', layout)
	pool = Pool(processes=nb_procs)
	
	
	while True:    
		event, vals = window.Read()     
		if event is None: break
		if event=='Close':
			window.Close()
			break
		if event=='USEDOM':
			window.Element('DOM').Update(disabled=(not vals['USEDOM']))
		if event=='Run':
			window.Hide()
			print('Start running...')
			print(' ')
			pars = {}
			
			for x,ax in enumerate(['MAX','MED','MIN']):
				pars[ax] = {}
				for p in ['LAG','NLAGS','BAND','ANGTOL','VBAND','VANGTOL']:
					if (ax=="MIN" and (p=='VBAND' or p=='VANGTOL')): continue
					if(p=='NLAGS'): pars[ax][p]=int(vals[p+str(x+1)])
					else: pars[ax][p]=float(vals[p+str(x+1)])
			
			pars['MIN']['VBAND']=pars['MIN']['BAND']
			pars['MIN']['VANGTOL']=pars['MIN']['ANGTOL']
			
			domvar = None
			if(vals['USEDOM']==True): domvar = vals['DOM']
			
			
			dataset,coordx,coordy,coordz,azm_code,dip_code,rak_code,vvar,outname,dom_code = tuple((vals['DATA'],vals['X'],vals['Y'],vals['Z'],vals['AZ'],vals['DP'],vals['RK'],vals['VVAR'],vals['OUT'],domvar))
			
			
			

			# Reading Dataset
			df_data=pd.read_csv(dataset,na_values='-')
			list_vars = [coordx,coordy,coordz,azm_code,dip_code,rak_code,vvar]
			if dom_code==None:
				df_data['DOM__'] = [1 for i in df_data.index.values]
				dom_code = 'DOM__'
			list_vars.append(dom_code)

			df_data.dropna(subset=list_vars,inplace=True)
			df_data = df_data[list_vars].reset_index(drop=True)
			df_data['IDX'] = range(len(df_data))
			#df_data.to_csv('idx.csv',index=False)

			out_df = pd.DataFrame(columns=['AXIS','BIN','NPAIRS','DIST','CORR'])
			
			idx_pair={}
			hlimit_d={}
			vlimit_d={}
			
			for ax in ['MAX','MED','MIN']:
				idx_pair[ax]=[]
				hlimit_d[ax] = pars[ax]['BAND']/np.tan(np.deg2rad(pars[ax]['ANGTOL']))
				vlimit_d[ax] = pars[ax]['VBAND']/np.tan(np.deg2rad(pars[ax]['VANGTOL']))
			
			
			for dom in df_data[dom_code].unique():
			
				df_full = df_data[df_data[dom_code]==dom].reset_index()
				if(len(df_data[dom_code].unique())>1): print('Domain:',dom)
				print(" - Nb of samples:",len(df_full))
				if(len(df_full)<10):
					print(" - Skipping: not enough information")
					continue
				chunk_size = int(np.ceil(df_full.shape[0]/nb_procs))
				if(chunk_size==0): chunk_size=1
				chunks = [df_full.iloc[i:i+chunk_size] for i in range(0, df_full.shape[0], chunk_size)]

				id_to_print = dict(zip([int(np.percentile(df_full.index.values, x)) for x in range(0,105,20)],range(0,105,20)))
				
				print('Progress: [', end =" ", flush=True)
				results = [pool.apply_async(func=lva.exp_vario_pairs, args=(chunk,df_full,id_to_print,pars,hlimit_d,vlimit_d,coordx,coordy,coordz,azm_code,dip_code,rak_code)) for chunk in chunks]
				output = [p.get() for p in results]
				print(']', flush=True)
				
				for x in output:
					for ax in ['MAX','MED','MIN']:
						idx_pair[ax]+=x[ax]
				
				
			# Organizing pairs and corresponding values and distances
			pairs = pd.Series(idx_pair['MAX']+idx_pair['MED']+idx_pair['MIN'])
			pairs.drop_duplicates(inplace=True)
			
			chunk_size = int(np.ceil(pairs.shape[0]/nb_procs))
			chunks = [pairs.iloc[i:i+chunk_size] for i in range(0, pairs.shape[0], chunk_size)]
			
			results = [pool.apply_async(func=lva.pairs_df, args=(chunk,df_data,coordx,coordy,coordz,vvar)) for chunk in chunks]
			output = [p.get() for p in results]
			
			df = pd.concat(output,ignore_index=True,sort=True)
			

			for ax in ['MAX','MED','MIN']:
				
				# ORGANIZING BINS
				bins = np.array([(i*pars[ax]['LAG']+pars[ax]['LAG']/2.0) for i in range(pars[ax]['NLAGS']+1)])
				df['BIN'] = np.digitize(df.D, bins, right=True)

				## to consider mirror pairs twice
				a = pd.Series(idx_pair[ax],dtype='object')
				b = a.drop_duplicates(keep=False)
				c = a[~a.isin(b)].unique().tolist()

				d = df[df['P12'].isin(c)]

				df2 = pd.concat([df,d],ignore_index=True,sort=True)
				df2 = df2[df2['BIN']<=pars[ax]['NLAGS']].reset_index()

				# Getting CORRELOGRAM for each lag
				for lg in df2['BIN'].unique():
					df_AX = df2[(df2['P12'].isin(idx_pair[ax])) & (df2['BIN']==lg)]
					if len(df_AX)==0: continue
					cov_matrix = np.cov(df_AX['H'],df_AX['T'],bias=True)
					corr_AX = cov_matrix[0][1] / np.sqrt(cov_matrix[0][0]*cov_matrix[1][1])
					dist_AX = np.average(df_AX['D'])
					npairs_AX = len(df_AX)
					out_df = out_df.append({'AXIS':ax, 'BIN':lg, 'NPAIRS':npairs_AX, 'DIST':dist_AX, 'CORR':corr_AX}, ignore_index=True)

			out_df.sort_values(['AXIS','BIN'],inplace=True)
			out_df.to_csv(outname+'.csv',index=False)
			
			
			window.UnHide()
	pool.close()
	pool.join()

