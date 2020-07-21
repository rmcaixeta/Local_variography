# -*- coding: utf-8 -*-

import PySimpleGUI as sg
import os
import lva_vario_funcs as lva

sg.theme('DarkGrey3')
sg.SetOptions( button_color=('black','#ff7636') )
dmdir = 'C:'



layout = [
	[sg.Text(' ')],
	[sg.Text('Dataset:', size=(20, 1)), sg.InputText(dmdir+'/dataset_pts.csv',key='DATA1'),sg.FileBrowse(initial_folder=dmdir,file_types=(("CSV Files", "*.csv"),))],
	[sg.Text('- Coordinate columns:', size=(20, 1)),sg.InputText('X',key='X1',size=(10, 1)),sg.InputText('Y',key='Y1',size=(10, 1)),sg.InputText('Z',key='Z1',size=(10, 1))],
	[sg.Text(' ')],
	[sg.Text('Angles file:', size=(20, 1)), sg.InputText(dmdir+'/angs_pts.csv',key='DATA2'),sg.FileBrowse(initial_folder=dmdir,file_types=(("CSV Files", "*.csv"),))],
	[sg.Text('- Local rotation columns:', size=(20, 1)),sg.InputText('TRDIPDIR',key='AZ',size=(10, 1)),sg.InputText('TRDIP',key='DP',size=(10, 1)),sg.InputText('RAKE',key='RK',size=(10, 1))],
	[sg.Text('- Coordinate columns:', size=(20, 1)),sg.InputText('XPT',key='X2',size=(10, 1)),sg.InputText('YPT',key='Y2',size=(10, 1)),sg.InputText('ZPT',key='Z2',size=(10, 1))],
	[sg.Text('- Invert dip sign*:', size=(20, 1)),sg.Checkbox(' ',key='neg_dip',default=False)],
	[sg.Text('* The script convention assumes GSLIB rotation, where negative dip points down',font='Default 9')],
	[sg.Text(' ')],
	[sg.Text('Output datafile:', size=(20, 1)), sg.InputText('dataset_angs',key='OUT')],
	[sg.Text(' ')],
	[sg.Button('Run'), sg.Button('Close')],
	[sg.Text(' ')]
]

window = sg.Window('Angs to Datafile', layout)

while True:    
	event, vals = window.Read()     
	if event is None: break
	if event=='Close':
		window.Close()
		break
	if event=='Run':
		window.Hide()
		print('Start running...')
		print(' ')
		
		lva.angs_to_data(vals['DATA1'],vals['X1'],vals['Y1'],vals['Z1'],vals['DATA2'],vals['X2'],vals['Y2'],vals['Z2'],vals['AZ'],vals['DP'],vals['RK'],vals['neg_dip'],vals['OUT'])
		window.UnHide()
		
		
