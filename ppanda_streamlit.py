import streamlit as st
import numpy as np
from glob import glob
from nustarpipeline import utils
import os, sys
import pyxmmsas as pysas
from astropy.io import fits as pf
import json
import papermill as pm
import re
import base64
import yaml 
import tarfile
from streamlit.logger import get_logger
from nustarpipeline import data_manipulation
try:
    import xspec
    have_xspec = True
except:
    have_xspec = False

from subprocess import Popen, PIPE, STDOUT

logger = get_logger(__name__)

if 'PROJECT_NAME' in os.environ:
    project_name=os.environ['PROJECT_NAME']
else:
    if os.path.isfile('.renku/metadata/project'):
        with open('.renku/metadata/project') as ff:
            project_dict = json.load(ff)
            #st.text(' '.join(project_dict.keys()))
            project_name = project_dict['name']
    else:
        dirs = glob(os.environ['HOME']+'/pp*')
        if len(dirs) == 0:
            project_name=os.getcwd().split('/')[-1]
        else:
            project_name = dirs[0].split('/')[-1]
        # warning = 'Warning: tentative project name determination: ' + project_name + '\n' + os.getcwd()
        # raise "Cannot determine the project name"


st.set_page_config(page_title='PPANDA', page_icon=':panda_face:', layout="wide", initial_sidebar_state="auto", 
                   menu_items={'Get help' : 'https://renkulab.io/projects/carlo.ferrigno/%s/' % project_name,
                               'About' : 'relative to the paper TBD',
                               'Report a Bug': 'mailto:carlo.ferrigno@unige.ch'})

# try:
#    st.warning(warning)
# except:
#    pass


def read_results(fname, input_dict={}, keys= []):
    #st.warning(f'Reading \"{fname}\"')
    if os.path.isfile(fname):
        with open(fname, 'r') as ff:
            try:
                src_res = json.load(ff)
            except Exception as e:
                st.warning(f'Exception {e} while reading {fname}')
                src_res = None
        if len(keys) >0 and src_res is not None:
            for kk in keys:
                input_dict.update({kk: src_res.get(kk, None)})
            return input_dict
        else:
            return src_res
    else:
        input_dict


def get_period_formatted(fname):
    if os.path.isfile(fname) is False:
        return ''
    x = np.loadtxt(fname, dtype=np.double)
    if len(x >= 6):
        format_str = pysas.get_format_string(x[4], x[5], x[5])
        return format_str % x[4] + '  $\\pm$ ' + format_str % x[5]
    else:
        return ''


def get_efold_file(ff):
    tt = ff.split('/')
    to_join = tt[0:-1] + ['obs_lc/ef_pipe_res.dat']
    return '/'.join(to_join)


def get_spectrum_file(ff):
    tt = ff.split('/')
    to_join = tt[0:-1] + ['obs_spec/FPMA_sr_rbn.pi']
    return '/'.join(to_join)


def get_nb_file(ff, nb_name='modelling_2023-05-25.ipynb'):
    tt = ff.split('/')
    to_join = tt[0:-1] + [nb_name]
    return '/'.join(to_join)


def get_yaml_file(ff, nb_name='model_para.yaml'):
    tt = ff.split('/')  
    to_join = tt[0:-1] + [nb_name]
    return '/'.join(to_join)


def log_subprocess_output(pipe):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        logger.info(line.decode()[0:-1])


def run(cmd):
    logger.info("------------------------------------------------------------------------------------------------\n")
    logger.info("**** running %s ****\n" % cmd)
    # out=subprocess.call(cmd, stdout=logger, stderr=logger, shell=True)
    process = Popen('export DYLD_LIBRARY_PATH=$HEADAS/lib;'+cmd, stdout=PIPE, 
                    stderr=STDOUT, shell=True)
    with process.stdout:
        log_subprocess_output(process.stdout)
    out = process.wait()  # 0 means success

    logger.info("------------------------------------------------------------------------------------------------\n")

    logger.info("Command '%s' finished with exit value %d" % (cmd, out))

    return out


def show_pdf_asimage(file_path, container=None, width=None):
    if not os.path.isfile(file_path):
        return
    
    png_image = file_path.replace('pdf', 'png')
    if not os.path.isfile(png_image) or (round(os.stat(png_image).st_mtime) < round(os.stat(file_path).st_mtime)):
        cmd = 'convert -define registry:temporary-path=/tmp -density 160 -trim -quality 80 %s %s' % (file_path, png_image)
        run(cmd)
    
    if width is None:
        
        if container is None:
            st.image(png_image, use_container_width=True)
        else:
            container.image(png_image, use_container_width=True)
    else:
        if container is None:
            st.image(png_image, width=width)
        else:
            container.image(png_image, width=width)


def show_pdf(file_path, width=700, height=510, container=None):
    if not os.path.isfile(file_path):
        return
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" type="application/pdf"></iframe>'
    if container is None:
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        container.markdown(pdf_display, unsafe_allow_html=True)


@st.cache_data
def get_spectra_information(fname):
    
    ff = pf.open(fname)
    src = ff[1].header['OBJECT']
    
    tmp = re.findall(r'\d{1}[m,p]\d{1}', src)
    if len(tmp) > 0:
        src = src.replace(tmp[0], tmp[0].replace('m', '-').replace('p', '+')).replace('_', ' ')
    
    cwd = os.getcwd()
    dirname = os.path.dirname(fname)
    # print(dirname)
    os.chdir(dirname)
    if have_xspec:
        pysas.epic_xspec_mcmc_fit(xspec, 'mod_base.xcm', 
                                  pn_spec="FPMA_sr_rbn.pi",
                                  mos1_spec="FPMB_sr_rbn.pi",
                                  mos2_spec="none",
                                  ignore_string=2 * ['**-3.0,70.0-**'],
                                  outputfiles_basename="gw-base-", 
                                  run_chain=False, compute_errors=False)
        xspec.AllModels.calcFlux("3 70.0")
        rate = xspec.AllData(1).rate
        flux = xspec.AllData(1).flux
    else:
        rate = [0]
        flux = [0]
                                
    os.chdir(cwd)
    
    res_dict = {
        'exposure': ff[1].header['EXPOSURE'],
        'tstart': ff[1].header['DATE-OBS'].replace('T', ' '),
        'tstop': ff[1].header['DATE-END'].replace('T', ' '),
        'rate': rate,
        'flux': flux
    }
    
    # print(ff[1].header)
    ff.close()
    return res_dict

# Starts app


if os.path.isdir('/home/jovyan/%s' % project_name):
    base_dir = '/home/jovyan/%s/' % project_name
else:
    base_dir = ''

result_files = glob(base_dir+'../sources/*/*/*_output_results.json')

# st.warning(' '.join(result_files))

master_dict = {}

for r_f in result_files:
    rr = read_results(r_f)
    # st.warning(json.dumps(rr))
    if rr is None:
        continue
    ss = rr['source']
    
    spectral_keys = ['exposure',
                    'tstart',
                    'tstop' ,
                    'rate' ,
                    'flux' ]
    
    source_dict = {}
    
    for kk in spectral_keys:
        if kk in rr:
            source_dict.update({kk: rr[kk]})
        elif kk.upper() in rr:
            source_dict.update({kk: rr[kk.upper()]})
        else:
            # st.warning('Key %s not found in %s' % (kk, r_f))
            if kk == 'rate' or kk == 'flux':
                source_dict.update({kk: [0.0, 0.0]})
            else:
                source_dict.update({kk: 0.0})
    
    source_dict.update({'path': get_nb_file(r_f, nb_name='')})    

    if ss not in master_dict:
        master_dict.update({ss: {rr['OBSID']: source_dict}})
    else:
        master_dict[ss].update({rr['OBSID']: source_dict})

    timing_res_file = source_dict['path']+'/obs_lc/ef_pipe_res.dat'
    timing_res_file2 = source_dict['path']+'/frequency_output.json'
    try:
        source_dict.update({'period': data_manipulation.get_period_formatted(fname=timing_res_file, raw=False, frequency=False, fname2=timing_res_file2, html=True)})
    # try:
    #    source_dict.update( {'period': rr['period_formatted']})
    except:
        st.warning(f'files \"{timing_res_file}\" and \"{timing_res_file2}\" do not contain timing results')
        source_dict.update( {'period': '1.0'})

# function to show additional plots
files_to_show = ['_posterior-low_corner.pdf', '_posterior-high_corner.pdf', \
        '_1rst_armonicpulsed_fitted.pdf', \
        '_1rst_armonic_posterior-low_corner.pdf', '_1rst_armonic_posterior-high_corner.pdf',\
        '_2nd_armonicpulsed_fitted.pdf', \
        '_2nd_armonic_posterior-low_corner.pdf', '_2nd_armonic_posterior-high_corner.pdf']
titles_to_show = ['PF low-energy corner plot', 'PF high-energy corner plot', \
        '1st harmonic fit',
        '1st harmonic low-energy corner plot', '1st harmonic high-energy corner plot',
        '2nd harmonic fit',
        '2nd harmonic low-energy corner plot', '2nd harmonic high-energy corner plot']

default_plots = ['1st harmonic fit', '2nd harmonic fit']


def show_additional_plots(options, titles=titles_to_show, files=files_to_show, container=None):

    if container is None:
        container = st
    if len(options) >0 :
        container.header('Additional plots')
    col1, col2 = container.columns(2)
    col_ind = 0
    columns = [col1, col2]
    for oo in options:
        for i, (f, t) in enumerate(zip(files, titles)):
            if oo == t:
                if 'fit' in oo:
                    print(oo)
                    show_pdf_asimage(path+'/figures/%s_%s' % (source.replace(' ', '_'), obsid) + f, container=columns[col_ind])
                    col_ind += 1

    for oo in options:
        # print(oo)
        for i, (f, t) in enumerate(zip(files, titles)):
            # print(t)
            if oo == t:
                if 'corner' in oo:
                    # print(oo)
                    col1, col2, col3 = container.columns([0.1, 5, 0.1])
                    show_pdf_asimage(path+'/figures/%s_%s' % (source.replace(' ', '_'), obsid) + f, container=col2)


# Function to recompute the notebook
def recompute(x, template_nb=base_dir+'model_2023-05-25.ipynb'):
    x.update({'run_xspec': have_xspec})
    # print(x)
    import papermill as pm
    from datetime import datetime
    now = datetime.now().isoformat(timespec='seconds')
    cwd = os.getcwd()
    # print(cwd)
    try:
        pm.execute_notebook(template_nb, path+'modeled_'+now+'.ipynb', parameters=x, cwd=path, log_output=True)
        for ff in glob(path+'figures/*.png'):
            os.remove(ff)
        return path+'modeled_'+now+'.ipynb'
    except Exception as e:
        st.error('Error in running the notebook')
        st.error(e)
        os.chdir(cwd)
        return None

# Necessary after running failing notebooks
# os.chdir(os.environ['HOME'])
if base_dir != '':
    os.chdir(base_dir)

# Styles
css_file = 'work/%s/ppanda_style.css' % project_name
if os.path.isfile(css_file):
    with open(css_file) as f_css:
        st.markdown(f'<style>{f_css.read()}</style>', 
                    unsafe_allow_html=True)
else:
    css_file = 'ppanda_style.css' 
    if os.path.isfile(css_file):
        with open(css_file) as f_css:
            st.markdown(f'<style>{f_css.read()}</style>', 
                        unsafe_allow_html=True)

# Title and subtitle
st.title('PPANDA')
st.subheader('Pulse Profiles of Accreting Neutron stars Deeply Analyzed\nC. Ferrigno, E. Ambrosi, A. D\'A&igrave;, D. K. Maniadakis\n\nRelated published papers: [Ferrigno et al. 2023](https://doi.org/10.1051/0004-6361/202347062), [D\'A&igrave; et al. 2025](https://doi.org/10.1051/0004-6361/202451469)')

# Selcts source and OBSID and get the relative path
with st.sidebar:
    source = st.selectbox("Select the source from this menu", sorted(master_dict.keys()))
    obsid = st.selectbox("Select the NuSTAR OBSID from this menu", sorted(master_dict[source].keys()))

path = master_dict[source][obsid]['path']


def src_html(source, obsid, res_dict):

    keys1 = ['tstart', 'tstop', 'exposure']
    keys2 = [ 'rate', 'flux', 'period', 'min_sn']

    out_str = '<table class="ppanda" align="center">\n'
    out_str += "<thead>\n<tr>\n"
    out_str += '<th>Source</th>\n'
    out_str += '<th>ObsID</th>\n'
    out_str += '<th>Tstart</th>\n'
    out_str += '<th>Tstop</th>\n'
    out_str += '<th>Exposure</th>\n'   
    out_str += '</tr>\n'
    out_str += '</thead>\n<tbody>\n'
    out_str += '<tr class="units">\n'
    out_str += '<td></td>\n'
    out_str += '<td></td>\n'
    out_str += '<td colspan=2>UTC</td>\n'
    
    out_str += '<td>ks</td>\n'
    out_str += '</tr>\n'
    out_str += '<tr>\n'
    out_str += '<td>%s</td>\n' % source
    out_str += '<td>%s</td>\n' % obsid

    for kk in keys1:
        if 'exposure' == kk:
            #print(res_dict[kk])
            out_str += '<td>%.1f</td>\n' % (float(res_dict[kk])/1e3)
        else:    
            out_str += '<td>%s</td>\n' % res_dict[kk]
        
    out_str += '</tr>\n'
    out_str += '</tbody>\n</table>\n'

    ###################################

    out_str += '<table class="ppanda", align="center">\n'
    out_str += "<thead>\n<tr>\n"
    out_str += '<th>Rate</th>\n'
    out_str += '<th>Flux</th>\n'
    out_str += '<th>Period</th>\n'
    # out_str += '<th>Min S/N</th>\n'
    out_str += '</tr>\n'
    out_str += '</thead>\n<tbody>\n'
    out_str += '<tr class="units">\n'
    out_str += '<td>cts/s (3-70 keV)</th>\n'
    out_str += '<td>10<sup>-9</sup>erg/s/cm<sup>-2</sup></th>\n'
    out_str += '<td>s</td>\n'
    # out_str += '<td></td>\n'
    out_str += '</tr>\n'
    out_str += '<tr>\n'
    
    for kk in keys2:
        if 'rate' == kk:
            out_str += '<td>%.2f</td>\n' % (float(res_dict[kk][0]))
        elif 'flux' == kk:
            out_str += '<td>%.2f</td>\n' % (float(res_dict[kk][0])*1e9)
        elif kk == 'min_sn':
            # out_str += '<td>%.0f</td>\n' % float(res_dict[kk])
            pass
        else:    
            out_str += '<td>%s</td>\n' % (res_dict[kk])
        
    out_str += '</tr>\n'
    out_str += '</tbody>\n</table>\n'


    return out_str

# Loads last-used parameter file
with open(path+'model_para.yaml') as yf:
    para_dict = yaml.safe_load(yf)

if para_dict is None:
    st.warning(path+'model_para.yaml does not contain a dictionary')
    para_dict = {}

if 'hm_steps' not in para_dict:
     para_dict['hm_steps'] = 6000

master_dict[source][obsid].update({'min_sn': para_dict.get('min_sn_matrix', 10.0)})

st.markdown(src_html(source, obsid, master_dict[source][obsid]), unsafe_allow_html=True)

options = st.sidebar.multiselect('Select the additional plots to show', titles_to_show, default=default_plots, key='additional', 
        help="Select plots to show", on_change=None, 
        args=None, kwargs=None, disabled=False, label_visibility="visible", max_selections=None)

with st.sidebar:
    st.markdown("Selection of parameters to recompute")
   
    n_gaussians = max(1, len(para_dict.get('forced_gaussian_centroids',[])))
    n_gaussians = int(st.number_input('How many Gaussians in the high-energy part?', min_value=0, max_value=4, value=int(n_gaussians), 
                                         help='This defines how many Gaussian we want to fit'))

    para_form = st.form('Paramters to recompute plots')
       
    nbins_options = np.array([8, 16, 32, 64], dtype=int)
    #TODO better to look for actual bins that are present
    
    nbins = int(para_dict.get('nbins', 32))
    
    index_option = np.where(nbins_options == nbins)[0]
    if len(index_option > 0):
        index_option = int(np.where(nbins_options == nbins)[0][0])
    else:
        index_option = 3
              
    para_dict['nbins'] = int(para_form.selectbox('number of bins in profile', nbins_options, index = index_option, 
                                             help='Selecting the number of phase bins'))
    para_dict['min_sn_matrix'] = para_form.number_input('Minimum S/N', min_value=1., max_value=100., value=float(para_dict.get('min_sn_matrix', 8.0)))
    para_dict['method_calc_rms'] = para_form.selectbox('Method to compute the pulsed fraction', ('adaptive', 'explicit_rms', 'counts', 'minmax', 'area'), index=0)
    noFe = False
    if 'noFe' in para_dict:
        noFe = para_dict['noFe']
    para_dict['noFe'] = para_form.checkbox('Do NOT include a Gaussian at 6.4 keV', value=noFe,key='noFe' ,help='If selected, it does not include a feature at the Iron energy')
    para_dict['e1_flex'] = para_form.number_input('Minimum energy to search for a break', min_value=0.0, max_value=100., 
                                                  value=float(para_dict.get('e1_flex', 10.0)), help='if > of max, no break')
    para_dict['e2_flex'] = para_form.number_input('Maximum energy to search for a break', min_value=0.0, max_value=100., 
                                                  value=float(para_dict.get('e2_flex', 20.0)), help='if < of min, no break')
    temp_para_dict={}
    temp_para_dict['forced_gaussian_centroids'] = []
    temp_para_dict['forced_gaussian_sigmas'] = []
    temp_para_dict['forced_gaussian_amplitudes'] = []
    if n_gaussians > 0:
        for i in range(n_gaussians):
            names = ['forced_gaussian_centroids', 'forced_gaussian_sigmas', 'forced_gaussian_amplitudes']
            short_names = ['centroid', 'Sigma', 'Amplitude']
            for kk, nn in zip(names, short_names):
                try:
                # print(para_dict[kk])
                    def_val = float(para_dict[kk][i])
                except:
                    if nn=='centroid':
                        def_val = 30.
                    elif nn == 'Sigma':
                        def_val = 5.
                    else:
                        def_val = -1.0
                temp_para_dict[kk].append(para_form.number_input('Initial %s of Gaussian #%d' % (nn, i+1), min_value=-100., max_value=100., value=def_val, 
                                                        help="The initial value for the fitting, limits are derived from it."))
    for kk, ii in temp_para_dict.items():
        para_dict.update({kk: ii})
    threshold_p_value = para_dict.get('threshold_p_value', 0.05)

    para_dict['threshold_p_value'] = para_form.number_input('Threshold p-value for the polynomial', min_value=1e-3, max_value=1., value=threshold_p_value, 
                                                            help='If the degree of the polynomial is not fixed, it is determined by the p-value of the fit. If the p-value is larger tha the threshold, the polynomial degree is increased.')

    poly_deg = para_dict.get('poly_deg', [-1,-1])
    poly_deg[0] = int(para_form.number_input('Degree of the polynomial for the low-energy part', min_value=-1, max_value=6, value=poly_deg[0], help='If -1, it is determined by the p-value of the fit'))
    poly_deg[1] = int(para_form.number_input('Degree of the polynomial for the high-energy part', min_value=-1, max_value=6, value=poly_deg[1], help='If -1, it is determined by the p-value of the fit'))
    para_dict['poly_deg'] = poly_deg

    n_high_bins_to_exclude = para_dict.get('n_high_bins_to_exclude',0)
    para_dict['n_high_bins_to_exclude'] = para_form.number_input('Number of high-energy bins to exclude', min_value=0,max_value=10, value=n_high_bins_to_exclude)

    def_val = para_dict.get('show_initial_energy_guess', False)

    para_dict['show_initial_energy_guess'] = para_form.checkbox('Show initial guess of centroid energies', def_val)
    dump_second_harmonic = True
    if 'dump_second_harmonic' in para_dict:
        dump_second_harmonic = para_dict['dump_second_harmonic']
    para_dict['dump_second_harmonic'] = para_form.checkbox('Show the fit of the second harmonic in table',dump_second_harmonic)

    para_dict['hm_steps'] = int(para_form.number_input('Length of the MCMC chains', min_value=0,max_value=10000, value=int(para_dict.get('hm_steps', 500)), help="This controls the length, use zero to prevent mcmc run, a small number (~500) for test, a large one (~5000) for the actual run"))

    para_dict['read_values'] = para_form.checkbox('Read products if existent',True, help='Uncheck to force re computation')
    recompute_flag = para_form.form_submit_button('Recompute with updated parameters', help='This will run the notebooks with input parameters. It might be LONG.')


#make a placeholder
placeholder = st.empty()

with placeholder.container() as main_container:
    main_container = st.container()
# Shows files

    #show_pdf(path+'/figures/%s_%s_summary_plot.pdf' % (source.replace(' ', '_'), obsid), container=main_container)
    show_pdf_asimage(path+'/figures/%s_%s_summary_plot.pdf' % (source.replace(' ', '_'), obsid), container=main_container)

    if os.path.isfile(path + 'figures/%s_%s_harms_table.html' % (source.replace(' ', '_'), obsid)):
        with open(path + 'figures/%s_%s_harms_table.html' % (source.replace(' ', '_'), obsid)) as hf:
            html_table = hf.read()
        
        main_container.markdown(html_table.replace('<table', '<table align="center" '), unsafe_allow_html=True)

                
    show_additional_plots(options, container=main_container)


    # Diplay of parameters
    main_container.markdown('We used these parameters')
    main_container.json(para_dict, expanded=False)

    #If matrices should be downloaded
    if main_container.button('Prepare download of Energy-Phase matrices', key='download', help='This allowsyou o download the energy phase matrices used for computation'):

        output_filename = para_dict['source'].replace(' ', '_').replace('+','p')+'_'+obsid+'_enphase_matrices.tar.gz'
        
        with tarfile.open(output_filename, "w:gz") as tar:
            for ff in glob(path+'obs_lc/*_ENPHASE_%03d.fits' % para_dict['nbins']):
                tar.add(ff, os.path.basename(ff))

        with open(output_filename, "rb") as file:
            btn = main_container.download_button(
                    label="Download the zipped archive of matrices",
                    data=file,
                    file_name=output_filename,
                    mime="application/gzip"
                )
        if btn:
            os.remove(output_filename)


if recompute_flag:
    placeholder.empty()
    with st.spinner('Computing the notebook: please wait'):
        output_nb = recompute(para_dict)
        #TBD a progress bar from papermill?
        st.session_state.output_nb = output_nb
        st.session_state.source = source

if 'output_nb' in st.session_state:
    if st.session_state.source == source:
        nb_name = st.session_state.output_nb
        if not isinstance(nb_name, str):
            st.error('Error in running the notebook')
        else:
            if os.path.isfile(st.session_state.output_nb):
                if not os.path.isfile(st.session_state.output_nb.replace('ipynb','html')):
                    run('jupyter nbconvert --to html %s' % st.session_state.output_nb)
                with open(st.session_state.output_nb, "rb") as f_ipynb:
                    btn = st.download_button(
                    label="Download the executed notebook",
                    data=f_ipynb,
                    file_name=os.path.basename(st.session_state.output_nb),
                    mime="application/x-ipynb+json"
                )
                if os.path.isfile(st.session_state.output_nb.replace('ipynb','html')):
                    with open(st.session_state.output_nb.replace('ipynb','html'), "rb") as f_html:
                        btn = st.download_button(
                        label="Download the executed notebook as html",
                        data=f_html,
                        file_name=os.path.basename(st.session_state.output_nb.replace('ipynb','html')),
                        mime="text/html"
                    )

    
st.button('Reload plots')
    
    #Display source properties

    


