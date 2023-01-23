#!/usr/bin/env python
# coding: utf-8

import os
import re
import time
import subprocess
import tempfile
import shutil
import asyncio
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import jinja2
from zipfile import ZipFile
from bs4 import BeautifulSoup
from pyppeteer import launch
from numpy.random import default_rng
from PyPDF2 import PdfReader
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

def load_files(examen):
  l = []
  for i,sec in enumerate(examen['secciones']):
    d = pd.read_excel(sec['archivo']).reset_index().rename(columns={'index':'Pos'}).set_index('Unique Id')
    d['Sec'] = sec['nombre']
    d['Ord'] = i + 1
    d['Pos'] = d['Pos'] + 1
    d['EsPadre'] = d['Total Points'] == 0.0
    d['Salto'] = False
    d['Blanca'] = False
    d['Ultimo'] = False
    # Renumerar la prueba sin contar a los padres
    d = d.join(d.loc[d['EsPadre']==False,'Pos'].rank().rename('Ord_y'))
    d['Ord_y'] = d['Ord_y'].fillna(0).astype(int)
    d['Ord'] = d['Ord_y']
    d = d.drop(columns='Ord_y')
    
    for s in sec['saltos']:
      d.loc[d['Ord']==s,'Salto'] = True
    for b in sec['blancas']:
      d.loc[d['Ord']==b,'Blanca'] = True
    d.loc[d.iloc[-1:].index,'Ultimo'] = True
    l.append(d)
  df = pd.concat(l)
  return df

def replace_equations(markup):
  if pd.isna(markup):
    return
  soup = BeautifulSoup(markup, 'html5lib')
  imgs = soup.find_all(class_='Wirisformula')
  for img in imgs:
    mathml = img['data-mathml'].replace('«','««').replace('»','»»')
    img.replace_with(mathml)
  result = str(soup.body)[6:-7] # quitar <body></body> solo conservar lo de adentro
  result = result.replace('««','<').replace('»»','>').replace('¨','"').replace('§','&')
  result = result.replace('<math ','<math display="block"')
  matches = re.findall(r'<annotation encoding="LaTeX">.*?</annotation>',result,flags=re.DOTALL)
  for m in matches:
    result = result.replace(m,'')
  result = result.replace('<semantics>','').replace('</semantics>','')
  result = result.replace('\xa0',' ').replace(r'<p> </p>',' ')
  return result

def fix_images(markup):
  if pd.isna(markup):
    return
  soup = BeautifulSoup(markup, 'html5lib')
  imgs = soup.find_all('img')
  for img in imgs:
    img['src'] = 'https://app.fasttestweb.com' + img['src']
  return str(soup.body)[6:-7]

def process_items(examen,df):
  #agregar numero de texto a los textos
  df = df.join(df.loc[df['EsPadre'],'Pos'].rank().astype(int).astype(str).rename('numtext'))

  # quitar algunos estilos
  df['Item Text'] = df['Item Text'].str.replace('style="font-family: \'times new roman\', times;"','')
  df['Item Text'] = df['Item Text'].str.replace('style="font-size: 12pt;"','')
  df['Item Text'] = df['Item Text'].str.replace('style="text-align: justify; font-size: 12pt;"','style="text-align: justify;"')

  # Reemplazar las acuaciones a MathML
  df['Item Text'] = df.apply(lambda x: replace_equations(x['Item Text']),axis=1)
  df['Answer 1'] = df.apply(lambda x: replace_equations(x['Answer 1']),axis=1)
  df['Answer 2'] = df.apply(lambda x: replace_equations(x['Answer 2']),axis=1)
  df['Answer 3'] = df.apply(lambda x: replace_equations(x['Answer 3']),axis=1)
  df['Answer 4'] = df.apply(lambda x: replace_equations(x['Answer 4']),axis=1)

  # Reemplazar las rutas de las imagenes
  df['Item Text'] = df.apply(lambda x: fix_images(x['Item Text']),axis=1)
  df['Answer 1'] = df.apply(lambda x: fix_images(x['Answer 1']),axis=1)
  df['Answer 2'] = df.apply(lambda x: fix_images(x['Answer 2']),axis=1)
  df['Answer 3'] = df.apply(lambda x: fix_images(x['Answer 3']),axis=1)
  df['Answer 4'] = df.apply(lambda x: fix_images(x['Answer 4']),axis=1)

  # generar el orden de las alternativas
  rng = default_rng(examen['código'])
  op = np.arange(1,5)
  df['orden'] = df.apply(lambda x: rng.permutation(op) if x['Alternativas en enunciado']!=True else op,axis=1)
  # calcular la nueva "clave"
  df['clave'] = df['orden'].apply(lambda x: np.nonzero(x ==1)[0][0] +1)

  return df

def render(item_tpl,item,examen):
  return item_tpl.render(
    description = item['Item Text'],
    answer1 = item[f'Answer {item["orden"][0]}'], #segun el vector generado cual debería ser la primera alternativa
    answer2 = item[f'Answer {item["orden"][1]}'],
    answer3 = item[f'Answer {item["orden"][2]}'],
    answer4 = item[f'Answer {item["orden"][3]}'],
    padre = item['EsPadre'],
    num_texto = item['numtext'],
    salto = item['Salto'],
    blanca = item['Blanca'],
    resaltar_clave = examen['resaltar_clave'],
    clave = item['clave'],
    ocultar_alternativas = item['Alternativas en enunciado'],
  )

async def generate_content(examen,df):
  prueba_tpl = jinja_env.get_template('test.tpl.html')
  n = 1
  page = await browser.newPage()

  for i,sec in enumerate(examen['secciones']):
    d = df.loc[df['Sec']==sec['nombre'],:]
    body = '\n'.join(d['html'])
    end = n + d[d['EsPadre']==False].shape[0] -1
    last = True if i == (len(examen['secciones'])-1) else False
    prueba = prueba_tpl.render(nombre=sec['nombre'],items=body,start=n,end = end, tiempo = sec['tiempo'],last=last)
    n = n + d[d['EsPadre']==False].shape[0]
    with open(f"{pwd.name}/{sec['nombre']}.html",'w') as f:
      f.write(prueba)
    await page.goto(f"file://{pwd.name}/{sec['nombre']}.html",waitUntil='networkidle2')
    #time.sleep(5)
    await page.pdf({'path':f"{pwd.name}/{sec['nombre']}.pdf",'printBackground':True, 'format':'A4'}) 

  await page.close()


async def generate_background(fname,num_pages=2,start_page=1,sec_num=1,sec_name=""):
  global jinja_env
  global browser
  global pwd
  tpl = jinja_env.get_template('background.tpl.html')
  numchars = len(sec_name)
  if numchars < 22:
    namesize = 2.5
  else:
    namesize = 2.5 *(1-((numchars-21)/(numchars-1)))
  html = tpl.render(num_pages=num_pages,start_page=start_page,sec_num=sec_num,sec_name=sec_name,size=namesize)
  with open(f'{pwd.name}/{fname}.html','w') as f:
    f.write(html)
  page = await browser.newPage()
  await page.goto(f"file://{pwd.name}/{fname}.html",waitUntil='networkidle2')
  await page.pdf({'path':f'{pwd.name}/{fname}.pdf','printBackground':True, 'format':'A4'})
  await page.close()

async def generate_final_pdf(examen,start_page=2):
  global pwd
  secciones = []
  for i,sec in enumerate(examen['secciones']):
    reader = PdfReader(f"{pwd.name}/{sec['nombre']}.pdf")
    num_pages = len(reader.pages)
    await generate_background(f"{sec['nombre']}-background",num_pages=num_pages,start_page=start_page,sec_num=i+1,sec_name=sec['nombre'])
    start_page = start_page + num_pages
    
  for i,sec in enumerate(examen['secciones']):
    outname = f"{pwd.name}/{sec['nombre']}-{examen['código']}.pdf"
    subprocess.call(f"pdftk \"{pwd.name}/{sec['nombre']}.pdf\" multibackground \"{pwd.name}/{sec['nombre']}-background.pdf\" output \"{outname}\"",shell=True)
    secciones.append(outname)

  total = ' '.join([f"\"{pwd.name}/{sec['nombre']}-{examen['código']}.pdf\"" for sec in examen['secciones']])
  outname = f"{pwd.name}/PRUEBA-{examen['versión']}-{examen['código']}.pdf"
  subprocess.call(f"pdftk {total} cat output \"{outname}\"",shell=True)
  return [outname] + secciones

def generate_anskey(examen,df):
  global pwd
  items = df[~df['EsPadre']]
  claves = items['clave'].apply(
    lambda x: chr(64+x)
  )
  
  claves = np.where(items['Alternativas en enunciado']==True,items['Answer 1'].str[3:-4],claves)
  pd.DataFrame(claves,columns=[examen['versión']]).to_excel(
    f"{pwd.name}/CLAVE-{examen['versión']}-{examen['código']}.xlsx",
    index=False
  )
  return f"{pwd.name}/CLAVE-{examen['versión']}-{examen['código']}.xlsx"

jinja_env = jinja2.Environment(
  #donde están los templates, por defecto es la carpeta actual
  loader = jinja2.FileSystemLoader('templates'),autoescape= True
)

#copiar los assets a un directorio temporal
pwd = tempfile.TemporaryDirectory()

browser = None

async def generate():
  global jinja_env
  global pwd
  global browser
  global examen

  for file in os.listdir('assets'):
    shutil.copy(f'assets/{file}',f'{pwd.name}/{file}')

  df = load_files(examen)
  df = process_items(examen,df)

  item_tpl = jinja_env.get_template('item.tpl.html')
  df['html'] = df.apply(lambda x: render(item_tpl,x,examen),axis=1)

  # Objetos para convertir un html a pdf usando chromium
  browser = await launch({
      'executablePath':'/usr/bin/google-chrome-stable',
      'headless':True,
      'args':['--no-sandbox','--disable-setuid-sandbox']
    },handleSIGINT=False,
    handleSIGTERM=False,
    handleSIGHUP=False)

  await generate_content(examen,df)
  rutas = await generate_final_pdf(examen)
  ruta_clave = generate_anskey(examen,df)
  await browser.close()
  ruta_zip = f"{pwd.name}/{examen['versión']}-{examen['código']}.zip"
  with ZipFile(ruta_zip,'w') as z:
    for ruta_final in rutas:
      z.write(ruta_final,arcname=ruta_final.split('/')[-1])
    z.write(ruta_clave,arcname=ruta_clave.split('/')[-1])
  return ruta_zip

# Streamlit - para generar la "estructura" de la prueba

st.title('Diagramar prueba - FastTestWeb')

datos = st.container()

examen = {
  'versión': datos.text_input('Versión',help='Es solo para el nombre del archivo'),
  'código' : datos.number_input('Código',value=0,format='%d',help='Dejar en 0 si se genera por primera vez, ingresar un código si se desea mantener siempre la mismas claves'),
  'resaltar_clave': datos.checkbox('Resaltar clave',help='Resalta la clave en amarillo para la revisión'),
  'nsecciones':datos.number_input('Número de secciones',value=3,format='%d'),
  'secciones': []
}

if examen['código'] == 0:
  examen['código'] = int(time.time())

for i in range(examen['nsecciones']):
  container = st.container()
  container.subheader(f'Sección {i+1}')
  sec = {
    'archivo': container.file_uploader(f'Archivo {i+1}',help='El archivo de metadata exportado por FastTestWeb'),
    'nombre': container.text_input(f'Nombre {i+1}',help='Esta etiqueta sale en la primera página y tambien en los bordes'),
    'tiempo': container.text_input(f'Tiempo {i+1}',help='Esta etiqueta sale en la primera página'),
    'saltos': container.text_input(f'Saltos de página {i+1}',help='Indicar el número de ítem después del cual se quiere insertar un salto de página, separar por comas si se quiere indicar varios ej. 5,6,7'),
    'blancas': container.text_input(f'Páginas en blanco {i+1}',help='Indicar el número de ítem después del cual se quiere insertar una página en blanco, separar por comas si se quiere indicar varios ej. 5,6,7'),
  }
  sec['saltos'] = [int(i) for i in sec['saltos'].split(',') if sec['saltos']!='']
  sec['blancas'] = [int(i) for i in sec['blancas'].split(',') if sec['blancas']!='']
  examen['secciones'].append(sec)

submit = st.container()
resultados = st.container()
resultados.empty()

def procesar():
  global resultados
  with resultados:
    with st.spinner('Generando archivos...'):
      ruta_zip = asyncio.run(generate())
      st.header("Archivos generados")
      with open(ruta_zip,'rb') as file:
        st.download_button("Descargar Archivos",data=file,file_name=ruta_zip.split('/')[-1],mime="application/zip")

btn = submit.button('PROCESAR',on_click=procesar)
