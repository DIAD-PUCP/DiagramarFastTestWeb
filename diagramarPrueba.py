#!/usr/bin/env python
# coding: utf-8

from io import BytesIO
import os
import re
import time
import base64
import tempfile
import shutil
from typing import Annotated, Optional
import warnings
from zipfile import ZipFile
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, computed_field, field_serializer
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

def validar_saltos(saltos:str) -> list[str]:
    if saltos != '':
        return saltos.split(',')
    return []

class Seccion(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    archivo: Optional[BytesIO] = Field(default=None, exclude=True)
    nombre: str = ''
    tiempo: str = ''
    saltos: Annotated[list[str], BeforeValidator(validar_saltos)] = []
    derCuad: bool = False

    @field_serializer('saltos')
    def serialize_saltos(self,saltos:list[str]):
        return ','.join(saltos) if saltos else ''


class Examen(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    version: Optional[str] = None
    codigo: int = 0
    password: Optional[str] = None
    caratula: Optional[BytesIO] = Field(default=None, exclude=True)
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

def html2pdf2(html: str, browser: Optional[webdriver.Chrome] = None, wait_for_id: Optional[str] = None) -> BytesIO:
    if not browser:
        browser = get_browser()
    
    browser.get("data:text/html;base64," + base64.b64encode(html.encode('utf-8')).decode())
    if wait_for_id:
        browser.find_element(By.ID, wait_for_id)

    print_options = PrintOptions()
    print_options.page_width = 21.0
    print_options.page_height = 29.7
    print_options.background = True

    base64code = browser.print_page(print_options)
    pdf_file = BytesIO(base64.b64decode(base64code))
    return pdf_file

def merge_pdf(files: list[str], fname: str, path: str = os.getcwd()) -> str:
    outname = f"{path}/{fname}"
    writer = PdfWriter()
    for pdf in files:
        reader = PdfReader(pdf)
        writer.append(reader)
    writer.write(outname)
    return outname

def merge_pdf2(files: list[BytesIO]) -> BytesIO:
    writer = PdfWriter()
    res = BytesIO()
    for pdf in files:
        reader = PdfReader(pdf)
        writer.append(reader)
    writer.write(res)
    return res


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

def stamp_pdf2(content: BytesIO, stamp: BytesIO) -> BytesIO:
    stamp_reader = PdfReader(stamp)
    content_reader = PdfReader(content)
    writer = PdfWriter()
    res = BytesIO()
    for c, s in zip(stamp_reader.pages, content_reader.pages):
        new_page = c
        mediabox = c.mediabox
        new_page.merge_page(s)
        new_page.mediabox = mediabox
        writer.add_page(new_page)
    writer.write(res)
    return res


def encrypt_pdf(fname: str, password: str) -> None:
    reader = PdfReader(fname)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(password)
    with open(fname, "wb") as f:
        writer.write(f)

def encrypt_pdf2(content: BytesIO, password: str) -> BytesIO:
    reader = PdfReader(content)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(password)
    res = BytesIO()
    writer.write(res)
    return res


def load_files(examen: Examen) -> pd.DataFrame:
    l = []
    for sec in examen.secciones:
        d = pd.read_excel(sec.archivo).reset_index().rename(
            columns={'index': 'Pos'})
        d = d.set_index('Unique Id')
        d['Pos'] = d['Pos'] + 1
        d['Sec'] = sec.nombre
        d['EsPadre'] = d['Total Points'] == 0
        d['Salto'] = False
        d['Continue'] = False
        d['Blanca'] = False
        d['Ultimo'] = False

        # Renumerar la prueba sin contar a los padres
        d = d.join(d.loc[~d['EsPadre'], 'Pos'].rank().rename('Ord'))
        d['Ord'] = d['Ord'].fillna(0).astype(int)

        for s in sec.saltos:
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
    rng = default_rng(examen.codigo)
    op = np.arange(1, 5)
    df['orden'] = df.apply(lambda x: rng.permutation(
        op) if not x['EsPadre'] else op, axis=1)
    # calcular la nueva "clave"
    df['clave'] = df['orden'].apply(lambda x: np.nonzero(x == 1)[0][0] + 1)
    return df


def generate_anskey(df:pd.DataFrame,version:str) -> BytesIO:
    items = df[~df['EsPadre']]
    claves = items['clave'].apply(
        lambda x: chr(64+x)
    )
    clave_file = BytesIO()
    claves.rename(version).to_excel(
        clave_file,
        index=False
    )
    return clave_file

def generate_estructura(df: pd.DataFrame) -> BytesIO:
    items = df[~df['EsPadre']]
    comp = pd.read_excel(
        'Temas.xlsx', sheet_name='Competencia', dtype=str).set_index('Bank')
    temas = pd.read_excel('Temas.xlsx', sheet_name='Equivalencia',
                          dtype=str).set_index('RUTA FASTTEST')
    temas = temas.rename(columns={'CODTEMA': 'Tema', 'CODSUBTEMA': 'SubTema'})
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
    archivo_est = BytesIO()
    est.to_excel(archivo_est)
    return archivo_est


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


def render_item(item_tpl: jinja2.Template, item: pd.Series, resaltar_clave: bool)->str:
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
        resaltar_clave=resaltar_clave,
        clave=item['clave'],
    )


def generate_sec_html(df: pd.DataFrame, i: int, sec: Seccion, tpl: jinja2.Template, start: int = 1, last: bool = False, extra_css: str = '') -> str:
    body = '\n'.join(df['html'])
    end = start + df[~df['EsPadre']].shape[0] - 1
    prueba = tpl.render(nombre=sec.nombre, num_seccion=i+1, items=body,
                        start=start, end=end, tiempo=sec.tiempo, last=last, extra_css=extra_css)
    return prueba


def calculate_breaks(html: str, df: pd.DataFrame, browser: Optional[webdriver.Chrome] = None, wait_for_id: Optional[str] = None):
    if not browser:
        browser = get_browser()

    browser.get("data:text/html;base64," + base64.b64encode(html.encode('utf-8')).decode())
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


def generate_content(examen: Examen, df: pd.DataFrame, item_tpl: jinja2.Template, examen_tpl: jinja2.Template, browser: Optional[webdriver.Chrome] = None, path: str = os.getcwd())->list[tuple[str,str,BytesIO]]:
    start = 1
    if not browser:
        browser = get_browser()
    df['html'] = df.apply(lambda x: render_item(item_tpl, x, examen.resaltar_clave), axis=1)
    res = []
    for i, sec in enumerate(examen.secciones):
        d = df.loc[df['Sec'] == sec.nombre, :].copy()
        last = (i == (len(examen.secciones)-1))
        html = generate_sec_html(d, i, sec, examen_tpl, start,
                          last, '' if examen.extra_css is None else examen.extra_css)
        if sec.derCuad:
            calculate_breaks(html, d, browser=browser,
                             wait_for_id="finished")
            d['html'] = d.apply(lambda x: render_item(
                item_tpl, x, examen.resaltar_clave), axis=1)
            html = generate_sec_html(d, i, sec, examen_tpl, start,
                              last, '' if examen.extra_css is None else examen.extra_css)
        start = start + d[~d['EsPadre']].shape[0]

        with open(f"{path}/{sec.nombre}.html", 'w',encoding='utf-8') as f:
            f.write(html)
        pdf = html2pdf2(html, browser=browser,
                 wait_for_id="finished")
        with open(f"{path}/{sec.nombre}", "wb") as f:
            f.write(pdf.getvalue())
        res.append((sec.nombre,html,pdf))
    return res


def generate_background_html(sec: Seccion, tpl: jinja2.Template, grid: str, sec_num: int = 1, start_page: int = 2, path: str = os.getcwd()):
    reader = PdfReader(f"{path}/{sec.nombre}.pdf")
    num_pages = len(reader.pages)
    numchars = len(sec.nombre)
    if numchars < 20:
        namesize = 2.5
    else:
        namesize = 2.5 * (1-((numchars-19)/(numchars-1)))
    html = tpl.render(num_pages=num_pages, start_page=start_page, sec_num=sec_num,
                      sec_name=sec.nombre, size=namesize, derCuad=sec.derCuad, grid=grid)
    with open(f'{path}/{sec.nombre}-background.html', 'w',encoding='utf-8') as f:
        f.write(html)
    return num_pages


def generate_backgrounds(examen: Examen, tpl: jinja2.Template, start_page: int = 2, browser: Optional[webdriver.Chrome] = None, path: str = os.getcwd()):
    if not browser:
        browser = get_browser()
    with open('assets/grid.svg',encoding='utf-8') as f:
        grid = f.read()
    bgrid = base64.b64encode(grid.encode('utf-8')).decode('utf-8')
    for i, sec in enumerate(examen.secciones):
        start_page += generate_background_html(
            sec, tpl, sec_num=i+1, start_page=start_page, grid=bgrid, path=path)
        with open(f"{path}/{sec.nombre}-background.html",'r',encoding='utf-8') as f:
            pdf = html2pdf2(f.read(),browser=browser,wait_for_id="finished")
            with open(f"{path}/{sec.nombre}-background.pdf", "wb") as f2:
                f2.write(pdf.getvalue())


def generate_sec_pdfs(examen: Examen, path: str = os.getcwd()):
    secciones = []
    for sec in examen.secciones:
        outname = stamp_pdf(
            content=f"{path}/{sec.nombre}.pdf",
            stamp=f"{path}/{sec.nombre}-background.pdf",
            fname=f"{sec.nombre}-{examen.version}-{examen.codigo}.pdf",
            path=path
        )
        secciones.append(outname)
    return secciones


def generar_configuracion_yaml(examen: Examen, path: str = os.getcwd())->str:
    config = yaml.dump(examen.model_dump(exclude_none=True), default_flow_style=False,
                       sort_keys=False, allow_unicode=True)
    outname = f"{path}/config.yml"
    with open(outname, 'w',encoding='utf-8') as f:
        f.write(config)
    return outname


def generate(examen: Examen, include_html: bool = False):
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
    estructura = generate_estructura(df)
    ruta_estructura = f"{pwd.name}/ESTRUCTURA-{examen.version}-{examen.codigo}.xlsx"
    with open(ruta_estructura,'wb') as f:
        f.write(estructura.getvalue())
    clave = generate_anskey(df, version=examen.version) # type: ignore
    ruta_clave = f'{pwd.name}/CLAVE-{examen.version}-{examen.codigo}.xlsx'
    with open(ruta_clave,'wb') as f:
        f.write(clave.getvalue())

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
    if examen.caratula:
        ruta_caratula = f"{pwd.name}/CARÁTULA-{examen.version}.pdf"
        with open(ruta_caratula, 'wb') as f:
            f.write(examen.caratula.getbuffer())
        rutas = [ruta_caratula] + rutas

    ruta_final = merge_pdf(
        rutas, f"PRUEBA-{examen.version}-{examen.codigo}.pdf", path=pwd.name)

    if (examen.password) and (examen.password != ''):
        for archivo in rutas + [ruta_final]:
            encrypt_pdf(archivo, examen.password)

    ruta_yaml = generar_configuracion_yaml(examen, path=pwd.name)

    rutas = rutas + [ruta_final, ruta_clave, ruta_estructura, ruta_yaml]

    if include_html:
        rutas = rutas + \
            [f"{pwd.name}/{r}" for r in os.listdir(pwd.name)
             if r.endswith('.html')]

    ruta_zip = f"{pwd.name}/{examen.version}-{examen.codigo}.zip"
    with ZipFile(ruta_zip, 'w') as z:
        for ruta_final in rutas:
            z.write(ruta_final, arcname=ruta_final.split('/')[-1])
    return ruta_zip, pwd


def load_yaml(obj: BytesIO)->Examen:
    yml = yaml.safe_load(obj)
    return Examen.model_validate(yml)


def main():
    # Streamlit - para generar la "estructura" de la prueba

    st.set_page_config(initial_sidebar_state='collapsed')
    st.title('Diagramar prueba - FastTestWeb')

    with st.sidebar:
        est_config = st.file_uploader(
            'Archivo de configuración',
            help='El archivo de configuración'
        )
        include_html = st.checkbox(
            'Incluir los Html generados en el comprimido',
            value=False,
            help='Los archivos HTML sirven para depurar errores'
        )
    
    if est_config:
        default_examen = load_yaml(est_config)
    else:
        default_examen = Examen()

    datos = st.container()

    examen = {
        'version': datos.text_input(
            'Version',
            value=default_examen.version,
            help='Es para el nombre de archivo y asignar los temas y subtemas en la estructura (CIENCIAS,LETRAS,ARTE)'
        ),
        'codigo': datos.number_input(
            'Código',
            value=default_examen.codigo,
            format='%d',
            help='Dejar en 0 si se genera por primera vez, ingresar un código si se desea mantener siempre la mismas claves'
        ),
        'caratula': datos.file_uploader(
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

    if examen['codigo'] == 0:
        examen['codigo'] = int(time.time())

    for i in range(examen['nsecciones']):
        if default_examen.nsecciones != 0:
            default_seccion = default_examen.secciones[i]
        else:
            default_seccion = Seccion()
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
                value=','.join(default_seccion.saltos),
                help='Indicar el número de ítem después del cual se quiere insertar un salto de página, separar por comas si se quiere indicar varios ej. 5,6,7'
            ),
            'derCuad': container.checkbox(
                'La cara derecha es cuadriculada',
                value=default_seccion.derCuad,
                help='Si se desea que la cara derecha (abierto como libro) sea cuadriculada',
                key=f'Quad{i}'
            ),
        }
        examen['secciones'].append(sec)

    submit = st.container()
    resultados = st.container()
    resultados.empty()

    btn = submit.button('PROCESAR')
    if btn:
        with resultados:
            with st.spinner('Generando archivos...'):
                ruta_zip, _ = generate(Examen.model_validate(examen), include_html)
                st.header("Archivos generados")
                with open(ruta_zip, 'rb') as file:
                    st.download_button(
                        "Descargar Archivos",
                        data=file,
                        file_name=ruta_zip.split('/')[-1],
                        mime="application/zip"
                    )


if __name__ == "__main__":
    main()
