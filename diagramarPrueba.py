#!/usr/bin/env python
# coding: utf-8

from io import BytesIO
import os
import re
import time
import base64
import tempfile
import shutil
from typing import Optional
import warnings
from zipfile import ZipFile
from copy import deepcopy
from pydantic import BaseModel, ConfigDict, computed_field
import yaml
import pandas as pd
import numpy as np
import streamlit as st
import jinja2
from bs4 import BeautifulSoup
from numpy.random import default_rng
from pypdf import PdfReader, PdfWriter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.print_page_options import PrintOptions
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


class Seccion(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    archivo: Optional[BytesIO] = None
    nombre: str = ''
    tiempo: str = ''
    saltos: list[str] = []
    derCuad: bool = False


class Examen(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    version: Optional[str] = None
    codigo: int = 0
    password: Optional[str] = None
    caratula: Optional[BytesIO] = None
    resaltar_clave: bool = False
    secciones: list[Seccion] = []
    extra_css: Optional[str] = None

    @computed_field
    def nsecciones(self) -> int:
        return len(self.secciones)


def get_browser() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-setuid-sandbox")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(2)
    return driver


def html2pdf(file: str, browser: Optional[webdriver.Chrome] = None, wait_for_id: Optional[str] = None, path: str = os.getcwd()) -> None:
    fname, extension = file.rsplit('.', 1)
    if extension != "html":
        raise ValueError(f"File {file} must be an html file")

    if not browser:
        browser = get_browser()

    browser.get(f"file://{path}/{fname}.html")
    if wait_for_id:
        browser.find_element(By.ID, wait_for_id)

    print_options = PrintOptions()
    print_options.page_width = 21.0
    print_options.page_height = 29.7
    print_options.background = True

    base64code = browser.print_page(print_options)
    with open(f"{path}/{fname}.pdf", "wb") as f:
        f.write(base64.b64decode(base64code))


def merge_pdf(files: list[str], fname: str, path: str = os.getcwd()) -> str:
    outname = f"{path}/{fname}"
    writer = PdfWriter()
    for pdf in files:
        reader = PdfReader(pdf)
        writer.append(reader)
    writer.write(outname)
    return outname


def stamp_pdf(content: str, stamp: str, fname: str, path: str = os.getcwd()) -> str:
    outname = f"{path}/{fname}"
    stamp_reader = PdfReader(stamp)
    content_reader = PdfReader(content)
    writer = PdfWriter()
    for c, s in zip(stamp_reader.pages, content_reader.pages):
        new_page = c
        mediabox = c.mediabox
        new_page.merge_page(s)
        new_page.mediabox = mediabox
        writer.add_page(new_page)
    writer.write(outname)
    return outname


def encrypt_pdf(fname: str, password: str) -> None:
    reader = PdfReader(fname)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(password)
    with open(fname, "wb") as f:
        writer.write(f)


def load_files(examen: dict) -> pd.DataFrame:
    l = []
    for _, sec in enumerate(examen['secciones']):
        d = pd.read_excel(sec['archivo']).reset_index().rename(
            columns={'index': 'Pos'})
        d = d.set_index('Unique Id')
        d['Pos'] = d['Pos'] + 1
        d['Sec'] = sec['nombre']
        d['EsPadre'] = d['Total Points'] == 0
        d['Salto'] = False
        d['Continue'] = False
        d['Blanca'] = False
        d['Ultimo'] = False

        # Renumerar la prueba sin contar a los padres
        d = d.join(d.loc[d['EsPadre'] == False, 'Pos'].rank().rename('Ord'))
        d['Ord'] = d['Ord'].fillna(0).astype(int)

        for s in sec['saltos']:
            if s[-1] == '*':
                s = int(s[:-1])
                d.loc[d['Ord'] == s, 'Continue'] = True
            else:
                s = int(s)
            d.loc[d['Ord'] == s, 'Salto'] = True

        d.loc[d.iloc[-1:].index, 'Ultimo'] = True

        l.append(d)

    df = pd.concat(l)

    # agregar numero de texto a los textos
    df = df.join(df.loc[df['EsPadre'], 'Pos'].rank().astype(
        int).astype(str).rename('numtext'))

    # generar el orden de las alternativas
    rng = default_rng(examen['código'])
    op = np.arange(1, 5)
    df['orden'] = df.apply(lambda x: rng.permutation(
        op) if x['EsPadre'] != True else op, axis=1)
    # calcular la nueva "clave"
    df['clave'] = df['orden'].apply(lambda x: np.nonzero(x == 1)[0][0] + 1)
    return df


def generate_anskey(examen: dict, df: pd.DataFrame, path: str = os.getcwd()) -> str:
    items = df[~df['EsPadre']]
    claves = items['clave'].apply(
        lambda x: chr(64+x)
    )
    name = f"{path}/CLAVE-{examen['versión']}-{examen['código']}.xlsx"
    claves.rename(examen['versión']).to_excel(
        name,
        index=False
    )
    return name


def generate_estructura(examen: dict, df: pd.DataFrame, path: str = os.getcwd()) -> str:
    items = df[~df['EsPadre']]
    comp = pd.read_excel(
        'Temas.xlsx', sheet_name='Competencia', dtype=str).set_index('Bank')
    temas = pd.read_excel('Temas.xlsx', sheet_name='Equivalencia',
                          dtype=str).set_index('RUTA FASTTEST')
    temas = temas.rename(columns={'CODTEMA': 'Tema', 'CODSUBTEMA': 'SubTema'})
    ruta = f"{path}/ESTRUCTURA-{examen['versión']}-{examen['código']}.xlsx"
    est = items
    est = est.join(comp, on='Bank')
    est['Categoría'] = '02'
    est['Error'] = 0.05
    est = est.join(temas, on='Category Path')
    est['Posición'] = np.arange(est.shape[0])+1
    est = est[['Item Name', 'Competencia', 'Tema', 'SubTema',
               'Categoría', 'Stat 3', 'IRT b', 'Error', 'Posición', 'orden']]
    est.columns = ['CodPregunta OCA', 'Competencia', 'Tema', 'SubTema', 'Categoria',
                   'N', 'Medición', 'Error', 'Posición\npregunta', 'Orden Alternativas']
    est.to_excel(ruta)
    return ruta


def replace_equations(markup: str) -> Optional[str]:
    if pd.isna(markup):
        return
    soup = BeautifulSoup(markup, 'html5lib')
    imgs = soup.find_all(class_='Wirisformula')
    for img in imgs:
        mathml = img['data-mathml'].replace('«', '««').replace('»', '»»') # type: ignore
        img.replace_with(mathml) # type: ignore
    # quitar <body></body> solo conservar lo de adentro
    result = str(soup.body)[6:-7]
    result = result.replace('««', '<').replace(
        '»»', '>').replace('¨', '"').replace('§', '&')
    result = result.replace('<math ', '<math display="block"')
    matches = re.findall(
        r'<annotation encoding="LaTeX">.*?</annotation>', result, flags=re.DOTALL)
    for m in matches:
        result = result.replace(m, '')
    result = result.replace('<semantics>', '').replace('</semantics>', '')
    result = result.replace('\xa0', ' ').replace(
        r'<p> </p>', ' ').replace(r'<p style="text-align: justify;"> </p>', ' ')
    return result


def fix_images(markup: str) -> Optional[str]:
    if pd.isna(markup):
        return
    soup = BeautifulSoup(markup, 'html5lib')
    imgs = soup.find_all('img')
    for img in imgs:
        if img['src'].startswith('/ftw/PR?'): # type: ignore
            img['src'] = 'https://app.fasttestweb.com' + img['src'] # type: ignore
    return str(soup.body)[6:-7]


def process_items(df: pd.DataFrame) -> pd.DataFrame:
    # quitar algunos estilos
    df['Item Text'] = df['Item Text'].str.replace(
        'style="font-family: \'times new roman\', times;"', '')
    df['Item Text'] = df['Item Text'].str.replace(
        'style="font-size: 12pt;"', '')
    df['Item Text'] = df['Item Text'].str.replace(
        'style="text-align: justify; font-size: 12pt;"', 'style="text-align: justify;"')

    # Reemplazar las acuaciones a MathML
    df['Item Text'] = df['Item Text'].apply(replace_equations)
    df['Answer 1'] = df['Answer 1'].apply(replace_equations)
    df['Answer 2'] = df['Answer 2'].apply(replace_equations)
    df['Answer 3'] = df['Answer 3'].apply(replace_equations)
    df['Answer 4'] = df['Answer 4'].apply(replace_equations)

    # Reemplazar las rutas de las imagenes
    df['Item Text'] = df['Item Text'].apply(fix_images)
    df['Answer 1'] = df['Answer 1'].apply(fix_images)
    df['Answer 2'] = df['Answer 2'].apply(fix_images)
    df['Answer 3'] = df['Answer 3'].apply(fix_images)
    df['Answer 4'] = df['Answer 4'].apply(fix_images)

    return df


def render_item(item_tpl: jinja2.Template, item: pd.Series, examen: dict):
    return item_tpl.render(
        description=item['Item Text'],
        # segun el vector generado cual debería ser la primera alternativa
        answer1=item[f'Answer {item["orden"][0]}'],
        answer2=item[f'Answer {item["orden"][1]}'],
        answer3=item[f'Answer {item["orden"][2]}'],
        answer4=item[f'Answer {item["orden"][3]}'],
        num_item=item['Ord'],
        padre=item['EsPadre'],
        num_texto=item['numtext'],
        salto=item['Salto'],
        cont=item['Continue'],
        blanca=item['Blanca'],
        resaltar_clave=examen['resaltar_clave'],
        clave=item['clave'],
    )


def generate_sec_html(df: pd.DataFrame, i: int, sec: dict, tpl: jinja2.Template, start: int = 1, last: bool = False, extra_css: str = '', path: str = os.getcwd()) -> None:
    body = '\n'.join(df['html'])
    end = start + df[df['EsPadre'] == False].shape[0] - 1
    prueba = tpl.render(nombre=sec['nombre'], num_seccion=i+1, items=body,
                        start=start, end=end, tiempo=sec['tiempo'], last=last, extra_css=extra_css)
    with open(f"{path}/{sec['nombre']}.html", 'w',encoding='utf-8') as f:
        f.write(prueba)


def calculate_breaks(sec: dict, df: pd.DataFrame, browser: Optional[webdriver.Chrome] = None, wait_for_id: Optional[str] = None, path: str = os.getcwd()):
    if not browser:
        browser = get_browser()

    browser.get(f"file://{path}/{sec['nombre']}.html")
    if wait_for_id:
        browser.find_element(By.ID, wait_for_id)
    res = browser.execute_script('return getSizes()')
    total = res['titleSize']
    max_height = 1024
    for i, r in enumerate(res['itemSizes']):
        total += r
        if total > max_height:
            df.loc[df['Ord'] == i, 'Blanca'] = True
            total = r - res['itemMarginTop']/2


def generate_content(examen: dict, df: pd.DataFrame, item_tpl: jinja2.Template, examen_tpl: jinja2.Template, browser: Optional[webdriver.Chrome] = None, path: str = os.getcwd()):
    start = 1
    if not browser:
        browser = get_browser()
    df['html'] = df.apply(lambda x: render_item(item_tpl, x, examen), axis=1)
    for i, sec in enumerate(examen['secciones']):
        d = df.loc[df['Sec'] == sec['nombre'], :].copy()
        last = (i == (len(examen['secciones'])-1))
        generate_sec_html(d, i, sec, examen_tpl, start,
                          last, examen['extra_css'], path)
        if sec['derCuad']:
            calculate_breaks(sec, d, browser=browser,
                             wait_for_id="finished", path=path)
            d['html'] = d.apply(lambda x: render_item(
                item_tpl, x, examen), axis=1)
            generate_sec_html(d, i, sec, examen_tpl, start,
                              last, examen['extra_css'], path)
        start = start + d[d['EsPadre'] == False].shape[0]
        html2pdf(f"{sec['nombre']}.html", browser=browser,
                 wait_for_id="finished", path=path)


def generate_background_html(sec: dict, tpl: jinja2.Template, grid: str, sec_num: int = 1, start_page: int = 2, path: str = os.getcwd()):
    reader = PdfReader(f"{path}/{sec['nombre']}.pdf")
    num_pages = len(reader.pages)
    numchars = len(sec['nombre'])
    if numchars < 20:
        namesize = 2.5
    else:
        namesize = 2.5 * (1-((numchars-19)/(numchars-1)))
    html = tpl.render(num_pages=num_pages, start_page=start_page, sec_num=sec_num,
                      sec_name=sec['nombre'], size=namesize, derCuad=sec['derCuad'], grid=grid)
    with open(f'{path}/{sec["nombre"]}-background.html', 'w',encoding='utf-8') as f:
        f.write(html)
    return num_pages


def generate_backgrounds(examen: dict, tpl: jinja2.Template, start_page: int = 2, browser: Optional[webdriver.Chrome] = None, path: str = os.getcwd()):
    if not browser:
        browser = get_browser()
    with open('assets/grid.svg',encoding='utf-8') as f:
        grid = f.read()
    bgrid = base64.b64encode(grid.encode('utf-8')).decode('utf-8')
    for i, sec in enumerate(examen['secciones']):
        start_page += generate_background_html(
            sec, tpl, sec_num=i+1, start_page=start_page, grid=bgrid, path=path)
        html2pdf(f"{sec['nombre']}-background.html",
                 browser=browser, wait_for_id="finished", path=path)


def generate_sec_pdfs(examen: dict, path: str = os.getcwd()):
    secciones = []
    for _, sec in enumerate(examen['secciones']):
        outname = stamp_pdf(
            content=f"{path}/{sec['nombre']}.pdf",
            stamp=f"{path}/{sec['nombre']}-background.pdf",
            fname=f"{sec['nombre']}-{examen['versión']}-{examen['código']}.pdf",
            path=path
        )
        secciones.append(outname)
    return secciones


def generar_configuracion_yaml(examen: dict, path: str = os.getcwd()):
    ex = deepcopy(examen)
    ex['carátula'] = ex['carátula'].name if ex['carátula'] else None
    d = {}
    for sec in ex['secciones']:
        sec['saltos'] = ','.join(sec['saltos']) if sec['saltos'] else None
        sec['archivo'] = sec['archivo'].name
        d[sec['nombre']] = sec
        sec.pop('nombre')
    ex['secciones'] = d
    config = yaml.dump(ex, default_flow_style=False,
                       sort_keys=False, allow_unicode=True)
    outname = f"{path}/config.yml"
    with open(outname, 'w',encoding='utf-8') as f:
        f.write(config)
    return outname


def generate(examen: dict, include_html: bool = False):
    jinja_env = jinja2.Environment(
        # donde están los templates, por defecto es la carpeta actual
        loader=jinja2.FileSystemLoader('templates'), autoescape=True
    )

    # copiar los assets a un directorio temporal
    pwd = tempfile.TemporaryDirectory()

    for file in os.listdir('assets'):
        shutil.copy(f'assets/{file}', f'{pwd.name}/{file}')

    df = load_files(examen)

    # generar archivo de estructura y archivo de claves
    ruta_estructura = generate_estructura(examen, df, path=pwd.name)
    ruta_clave = generate_anskey(examen, df, path=pwd.name)

    df = process_items(df)

    # Objetos para convertir un html a pdf usando chromium
    browser = get_browser()
    item_tpl = jinja_env.get_template('item.tpl.html')
    prueba_tpl = jinja_env.get_template('test.tpl.html')
    generate_content(examen, df, item_tpl, prueba_tpl,
                     browser=browser, path=pwd.name)

    background_tpl = jinja_env.get_template('background.tpl.html')
    generate_backgrounds(examen, background_tpl,
                         start_page=2, browser=browser, path=pwd.name)

    browser.quit()

    rutas = generate_sec_pdfs(examen, path=pwd.name)

    # Agregar carátula si se selecciono una
    if examen['carátula']:
        ruta_caratula = f"{pwd.name}/CARÁTULA-{examen['versión']}.pdf"
        with open(ruta_caratula, 'wb') as f:
            f.write(examen['carátula'].getbuffer())
        rutas = [ruta_caratula] + rutas

    ruta_final = merge_pdf(
        rutas, f"PRUEBA-{examen['versión']}-{examen['código']}.pdf", path=pwd.name)

    if ('password' in examen) and (examen['password'] != ''):
        for archivo in rutas + [ruta_final]:
            encrypt_pdf(archivo, examen['password'])

    ruta_yaml = generar_configuracion_yaml(examen, path=pwd.name)

    rutas = rutas + [ruta_final, ruta_clave, ruta_estructura, ruta_yaml]

    if include_html:
        rutas = rutas + \
            [f"{pwd.name}/{r}" for r in os.listdir(pwd.name)
             if r.endswith('.html')]

    ruta_zip = f"{pwd.name}/{examen['versión']}-{examen['código']}.zip"
    with ZipFile(ruta_zip, 'w') as z:
        for ruta_final in rutas:
            z.write(ruta_final, arcname=ruta_final.split('/')[-1])
    return ruta_zip, pwd


def procesar(resultados, examen: dict, include_html: bool):
    with resultados:
        with st.spinner('Generando archivos...'):
            ruta_zip, _ = generate(examen, include_html)
            st.header("Archivos generados")
            with open(ruta_zip, 'rb') as file:
                st.download_button(
                    "Descargar Archivos",
                    data=file,
                    file_name=ruta_zip.split('/')[-1],
                    mime="application/zip"
                )


def load_yaml(obj: BytesIO):
    if obj == None:
        with open('defaults.yml', 'r',encoding='utf-8') as f:
            yml = yaml.safe_load(f)
    else:
        yml = yaml.safe_load(obj)
    secciones = []
    yml['carátula'] = None
    for name, sec in yml['secciones'].items():
        sec['nombre'] = name
        sec['archivo'] = None
        sec['saltos'] = '' if sec['saltos'] is None else sec['saltos']
        secciones.append(sec)
    yml['secciones'] = secciones
    if 'password' not in yml:
        yml['password'] = ''
    return yml


def main():
    # Streamlit - para generar la "estructura" de la prueba

    st.set_page_config(initial_sidebar_state='collapsed')
    st.title('Diagramar prueba - FastTestWeb')

    with st.sidebar:
        _ = st.file_uploader(
            'Archivo de configuración',
            help='El archivo de configuración'
        )
        include_html = st.checkbox(
            'Incluir los Html generados en el comprimido',
            value=False,
            help='Los archivos HTML sirven para depurar errores'
        )

    default_examen = Examen()
    default_seccion = Seccion()

    datos = st.container()

    examen = {
        'versión': datos.text_input(
            'Versión',
            value=default_examen.version,
            help='Es para el nombre de archivo y asignar los temas y subtemas en la estructura (CIENCIAS,LETRAS,ARTE)'
        ),
        'código': datos.number_input(
            'Código',
            value=default_examen.codigo,
            format='%d',
            help='Dejar en 0 si se genera por primera vez, ingresar un código si se desea mantener siempre la mismas claves'
        ),
        'carátula': datos.file_uploader(
            'Carátula',
            help='Adjuntar una carátula si se desea incluir en la versión final'
        ),
        'resaltar_clave': datos.checkbox(
            'Resaltar clave',
            value=default_examen.resaltar_clave,
            help='Resalta la clave en amarillo para la revisión'
        ),
        'nsecciones': datos.number_input(
            'Número de secciones',
            value=2,
            format='%d'
        ),
        'secciones': []
    }
    extra_styles = datos.checkbox(
        'Agregar estilos adicionales',
        value=default_examen.extra_css is not None,
        help='Marcar si se desea agregar estilos adicionales al contenido de la prueba'
    )

    if extra_styles:
        examen['extra_css'] = datos.text_area(
            'CSS extra',
            value=default_examen.extra_css,
            help='Agregar CSS extra al contenido de la prueba'
        )

    usar_password = datos.checkbox(
        'Agregar un password a los PDFs',
        value=default_examen.password is not None,
        help='Marcar si se desea agregar una contraseña a todos los archivos PDF'
    )

    if usar_password:
        examen['password'] = datos.text_input(
            'Password',
            value=default_examen.password,
            help='Password para los PDFs',
            type='password'
        )

    if examen['código'] == 0:
        examen['código'] = int(time.time())

    for i in range(examen['nsecciones']):
        container = st.container()
        container.subheader(f'Sección {i+1}')
        sec = {
            'archivo': container.file_uploader(
                f'Archivo {i+1}',
                help='El archivo de metadata exportado por FastTestWeb'
            ),
            'nombre': container.text_input(
                f'Nombre {i+1}',
                help='Esta etiqueta sale en la primera página de la sección y tambien en los bordes',
                value=default_seccion.nombre,
            ),
            'tiempo': container.text_input(
                f'Tiempo {i+1}',
                value=default_seccion.tiempo,
                help='Esta etiqueta sale en la primera página'
            ),
            'saltos': container.text_input(
                f'Saltos de página {i+1}',
                value=default_seccion.saltos if default_seccion.saltos else '',
                help='Indicar el número de ítem después del cual se quiere insertar un salto de página, separar por comas si se quiere indicar varios ej. 5,6,7'
            ),
            'derCuad': container.checkbox(
                'La cara derecha es cuadriculada',
                value=default_seccion.derCuad,
                help='Si se desea que la cara derecha (abierto como libro) sea cuadriculada',
                key=f'Quad{i}'
            ),
        }
        sec['saltos'] = [i for i in sec['saltos'].split(
            ',') if sec['saltos'] != '']
        examen['secciones'].append(sec)

    Examen.model_validate(examen)

    submit = st.container()
    resultados = st.container()
    resultados.empty()

    btn = submit.button('PROCESAR')
    if btn:
        procesar(resultados, examen, include_html)


if __name__ == "__main__":
    main()
