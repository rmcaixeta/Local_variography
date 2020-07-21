# -*- coding: utf-8 -*-

import PySimpleGUI as sg
import os
import lva_vario_funcs as lva
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

sg.theme('DarkGrey3')
sg.SetOptions( button_color=('black','#ff7636') )
dmdir = 'C:'


def draw_figure(canvas, figure, loc=(0, 0)):
	canvas.delete('ALL')
	figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
	figure_canvas_agg.draw()
	figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
	return figure_canvas_agg



layout_pars = [
	[sg.Text(' ')],
	[sg.Text('Exp. Vario:', size=(10, 1)), sg.InputText(dmdir+'/exp_variography.csv',key='DATA',size=(40, 1)),sg.FileBrowse(initial_folder=dmdir,file_types=(("CSV Files", "*.csv"),))],
	[sg.Text(' ')],
	[sg.Text('Model type:', size=(10, 1)), sg.InputCombo(('Spherical', 'Exponential'), default_value='Spherical',key='MODTYP',readonly=True)],
	[sg.Text(' ')],
	[sg.Text('Nugget:', size=(20, 1)), sg.InputText('0.08',key='NGT',size=(10, 1))],
	[sg.Text('1st structure cc:', size=(20, 1)), sg.InputText('0.54',key='CC1',size=(10, 1))],
	[sg.Text('1st structure ranges:', size=(20, 1)),sg.InputText('39.9',key='R11',size=(10, 1)),sg.InputText('36.8',key='R12',size=(10, 1)),sg.InputText('0',key='R13',size=(10, 1))],
	[sg.Text('2nd structure cc:', size=(20, 1)), sg.InputText('0.38',key='CC2',size=(10, 1))],
	[sg.Text('2nd structure ranges:', size=(20, 1)),sg.InputText('132',key='R21',size=(10, 1)),sg.InputText('36.8',key='R22',size=(10, 1)),sg.InputText('0',key='R23',size=(10, 1))],
	[sg.Text('3rd structure cc:', size=(20, 1)), sg.InputText('0.0',key='CC3',size=(10, 1))],
	[sg.Text('3rd structure ranges:', size=(20, 1)),sg.InputText('0',key='R31',size=(10, 1)),sg.InputText('0',key='R32',size=(10, 1)),sg.InputText('0',key='R33',size=(10, 1))],
	[sg.Text(' ')],
	[sg.Text('Show pairs histogram:', size=(20, 1)),sg.Checkbox(' ',key='HIST',default=True)],
	[sg.Text(' ')],
	[sg.Button('Run'), sg.Button('Close')]
]


layout = [
	[sg.Column(layout_pars)]
]

window = sg.Window('LVA Vario', layout, finalize=True)

while True:    
	event, vals = window.Read()     
	if event is None: break
	if event=='Close':
		window.Close()
		break
	if event=='Run':
		
		par={}
		for x,ax in enumerate(['MAX','MED','MIN']):
			par[ax] = {}
			par[ax]['a1'] = float(vals['R1'+str(x+1)])
			par[ax]['a2'] = float(vals['R2'+str(x+1)])
			par[ax]['a3'] = float(vals['R3'+str(x+1)])

		window.Close()
		fig_parts = lva.model_vario(vals['DATA'],vals['MODTYP'],float(vals['NGT']),float(vals['CC1']),float(vals['CC2']),float(vals['CC3']),vals['HIST'],par)
				
		layout_pars = [
			[sg.Text(' ')],
			[sg.Text('Exp. Vario:', size=(10, 1)), sg.InputText(vals['DATA'],key='DATA',size=(40, 2)),sg.FileBrowse(initial_folder=dmdir,file_types=(("CSV Files", "*.csv"),))],
			[sg.Text(' ')],
			[sg.Text('Model type:', size=(10, 1)), sg.InputCombo(('Spherical', 'Exponential'), default_value=vals['MODTYP'],key='MODTYP',readonly=True)],
			[sg.Text(' ')],
			[sg.Text('Nugget:', size=(20, 1)), sg.InputText(vals['NGT'],key='NGT',size=(10, 1))],
			[sg.Text('1st structure cc:', size=(20, 1)), sg.InputText(vals['CC1'],key='CC1',size=(10, 1))],
			[sg.Text('1st structure ranges:', size=(20, 1)),sg.InputText(vals['R11'],key='R11',size=(10, 1)),sg.InputText(vals['R12'],key='R12',size=(10, 1)),sg.InputText(vals['R13'],key='R13',size=(10, 1))],
			[sg.Text('2nd structure cc:', size=(20, 1)), sg.InputText(vals['CC2'],key='CC2',size=(10, 1))],
			[sg.Text('2nd structure ranges:', size=(20, 1)),sg.InputText(vals['R21'],key='R21',size=(10, 1)),sg.InputText(vals['R22'],key='R22',size=(10, 1)),sg.InputText(vals['R23'],key='R23',size=(10, 1))],
			[sg.Text('3rd structure cc:', size=(20, 1)), sg.InputText(vals['CC3'],key='CC3',size=(10, 1))],
			[sg.Text('3rd structure ranges:', size=(20, 1)),sg.InputText(vals['R31'],key='R31',size=(10, 1)),sg.InputText(vals['R32'],key='R32',size=(10, 1)),sg.InputText(vals['R33'],key='R33',size=(10, 1))],
			[sg.Text(' ')],
			[sg.Text('Show pairs histogram:', size=(20, 1)),sg.Checkbox(' ',key='HIST',default=vals['HIST'])],
			[sg.Text(' ')],
			[sg.Button('Run'), sg.Button('Close')]
		]
		
		out_layout = [[sg.Canvas(size=(fig_parts[1], fig_parts[2]),key='canvas')]]
		layout = [[sg.Column(layout_pars),sg.Column(out_layout)]]
		
		window = sg.Window('LVA Vario', layout, finalize=True)
		#window = sg.Window('Model', out_layout, finalize=True)
		fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig_parts[0])
		window.UnHide()
		
		
