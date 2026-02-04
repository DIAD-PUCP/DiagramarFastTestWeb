#!/usr/bin/env python
# coding: utf-8

import base64
import re
import time
import warnings
from io import BytesIO
from typing import Annotated, Optional
from zipfile import ZipFile

import jinja2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from bs4 import BeautifulSoup
from numpy.random import default_rng
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_serializer
from pypdf import PdfReader, PdfWriter
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.print_page_options import PrintOptions

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def validar_saltos(saltos: str) -> list[str]:
    if saltos != "":
        return saltos.split(",")
    return []


class Seccion(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    archivo: Annotated[Optional[BytesIO], Field(exclude=True)] = None
    nombre: str = ""
    tiempo: str = ""
    saltos: Annotated[list[str], BeforeValidator(validar_saltos)] = []
    derCuad: bool = False
    items_df: Annotated[pd.DataFrame, Field(exclude=True)] = pd.DataFrame()
    html: Annotated[Optional[str], Field(exclude=True)] = None
    pdf_final: Annotated[Optional[BytesIO], Field(exclude=True)] = None
    pdf: Annotated[Optional[BytesIO], Field(exclude=True)] = None
    bhtml: Annotated[Optional[str], Field(exclude=True)] = None
    bpdf: Annotated[Optional[BytesIO], Field(exclude=True)] = None

    def num_pages(self) -> int:
        if self.pdf is not None:
            reader = PdfReader(self.pdf)
            num_pages = len(reader.pages)
            return num_pages
        return 0

    def name_size(self) -> float:
        numchars = len(self.nombre)
        if numchars < 20:
            namesize = 2.5
        else:
            namesize = 2.5 * (1 - ((numchars - 19) / (numchars - 1)))
        return namesize

    @field_serializer("saltos")
    def serialize_saltos(self, saltos: list[str]):
        return ",".join(saltos) if saltos else ""

    def generate_sec_html(
        self,
        tpl: jinja2.Template,
        num_seccion: int,
        styles: str,
        start: int = 1,
        last: bool = False,
        extra_css: Optional[str] = None,
    ):
        if not self.items_df.empty:
            body = "\n".join(self.items_df["html"])
            end = start + self.items_df[~self.items_df["EsPadre"]].shape[0] - 1
            html = tpl.render(
                nombre=self.nombre,
                num_seccion=num_seccion,
                items=body,
                start=start,
                end=end,
                tiempo=self.tiempo,
                last=last,
                extra_css=extra_css or "",
                styles=styles,
            )
            self.html = html

    def calculate_breaks(
        self,
        browser: Optional[webdriver.Chrome] = None,
        wait_for_id: Optional[str] = None,
    ):
        if (self.html is not None) and (not self.items_df.empty):
            if not browser:
                browser = get_browser()

            browser.get(
                "data:text/html;base64,"
                + base64.b64encode(self.html.encode("utf-8")).decode()
            )
            if wait_for_id:
                try:
                    browser.find_element(By.ID, wait_for_id)
                except NoSuchElementException:
                    time.sleep(5)
            res = browser.execute_script("return getSizes()")
            total = res["titleSize"]
            max_height = 1024
            for i, r in enumerate(res["itemSizes"]):
                total += r
                if total > max_height:
                    self.items_df.loc[self.items_df["Ord"] == i, "Blanca"] = True
                    total = r - res["itemMarginTop"] / 2

    def generate_background_html(
        self, tpl: jinja2.Template, grid: str, sec_num: int = 1, start_page: int = 2
    ):
        self.bhtml = tpl.render(
            num_pages=self.num_pages(),
            start_page=start_page,
            sec_num=sec_num,
            sec_name=self.nombre,
            size=self.name_size(),
            derCuad=self.derCuad,
            grid=grid,
        )


class Examen(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    version: Optional[str] = None
    codigo: int = 0
    password: Optional[str] = None
    caratula: Annotated[Optional[BytesIO], Field(exclude=True)] = None
    resaltar_clave: bool = False
    secciones: list[Seccion] = []
    extra_css: Optional[str] = None
    include_html: Annotated[bool, Field(exclude=True)] = False
    pdf: Annotated[Optional[BytesIO], Field(exclude=True)] = None
    clave: Annotated[Optional[BytesIO], Field(exclude=True)] = None
    estructura: Annotated[Optional[BytesIO], Field(exclude=True)] = None
    config_yaml: Annotated[Optional[str], Field(exclude=True)] = None

    def nsecciones(self) -> int:
        return len(self.secciones)

    def generar_configuracion_yaml(self):
        config = yaml.dump(
            self.model_dump(exclude_none=True),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        self.config_yaml = config


def get_browser() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-setuid-sandbox")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(4)
    return driver


def html2pdf(
    html: str,
    browser: Optional[webdriver.Chrome] = None,
    wait_for_id: Optional[str] = None,
) -> BytesIO:
    if not browser:
        browser = get_browser()
    browser.get(
        "data:text/html;base64," + base64.b64encode(html.encode("utf-8")).decode()
    )
    if wait_for_id:
        try:
            browser.find_element(By.ID, wait_for_id)
        except NoSuchElementException:
            time.sleep(5)

    print_options = PrintOptions()
    print_options.page_width = 21.0
    print_options.page_height = 29.7
    print_options.background = True

    base64code = browser.print_page(print_options)
    pdf_file = BytesIO(base64.b64decode(base64code))
    return pdf_file


def merge_pdf(files: list[BytesIO]) -> BytesIO:
    writer = PdfWriter()
    for pdf in files:
        reader = PdfReader(pdf)
        writer.append(reader)
    res = BytesIO()
    writer.write(res)
    return res


def stamp_pdf(content: BytesIO, stamp: BytesIO) -> BytesIO:
    stamp_reader = PdfReader(stamp)
    content_reader = PdfReader(content)
    writer = PdfWriter()
    for c, s in zip(stamp_reader.pages, content_reader.pages):
        new_page = c
        mediabox = c.mediabox
        new_page.merge_page(s)
        new_page.mediabox = mediabox
        writer.add_page(new_page)
    res = BytesIO()
    writer.write(res)
    return res


def encrypt_pdf(content: BytesIO, password: str) -> BytesIO:
    reader = PdfReader(content)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(password)
    res = BytesIO()
    writer.write(res)
    return res


def load_files(examen: Examen) -> pd.DataFrame:
    df_l = []
    for sec in examen.secciones:
        d = pd.read_excel(sec.archivo).reset_index().rename(columns={"index": "Pos"})
        d = d.set_index("Unique Id")
        d["Pos"] = d["Pos"] + 1
        d["Sec"] = sec.nombre
        d["EsPadre"] = d["Total Points"] == 0
        d["Salto"] = False
        d["Continue"] = False
        d["Blanca"] = False
        d["Ultimo"] = False

        # Renumerar la prueba sin contar a los padres
        d = d.join(d.loc[~d["EsPadre"], "Pos"].rank().rename("Ord"))
        d["Ord"] = d["Ord"].fillna(0).astype(int)

        for s in sec.saltos:
            if s[-1] == "*":
                s = int(s[:-1])
                d.loc[d["Ord"] == s, "Continue"] = True
            else:
                s = int(s)
            d.loc[d["Ord"] == s, "Salto"] = True

        d.loc[d.iloc[-1:].index, "Ultimo"] = True

        df_l.append(d)

    df = pd.concat(df_l)

    # agregar numero de texto a los textos
    df = df.join(
        df.loc[df["EsPadre"], "Pos"].rank().astype(int).astype(str).rename("numtext")
    )

    # generar el orden de las alternativas
    rng = default_rng(examen.codigo)
    op = np.arange(1, 5)
    df["orden"] = df.apply(
        lambda x: rng.permutation(op) if not x["EsPadre"] else op, axis=1
    )
    # calcular la nueva "clave"
    df["clave"] = df["orden"].apply(lambda x: np.nonzero(x == 1)[0][0] + 1)
    return df


def generate_anskey(df: pd.DataFrame, version: str) -> BytesIO:
    items = df[~df["EsPadre"]]
    claves = items["clave"].apply(lambda x: chr(64 + x))
    clave_file = BytesIO()
    claves.rename(version).to_excel(clave_file, index=False)
    return clave_file


def generate_estructura(df: pd.DataFrame) -> BytesIO:
    items = df[~df["EsPadre"]]
    comp = pd.read_excel("Temas.xlsx", sheet_name="Competencia", dtype=str).set_index(
        "Bank"
    )
    temas = pd.read_excel("Temas.xlsx", sheet_name="Equivalencia", dtype=str).set_index(
        "RUTA FASTTEST"
    )
    temas = temas.rename(columns={"CODTEMA": "Tema", "CODSUBTEMA": "SubTema"})
    est = items
    est = est.join(comp, on="Bank")
    est["Categoría"] = "02"
    est["Error"] = 0.05
    est = est.join(temas, on="Category Path")
    est["Posición"] = np.arange(est.shape[0]) + 1
    est = est[
        [
            "Item Name",
            "Competencia",
            "Tema",
            "SubTema",
            "Categoría",
            "Stat 3",
            "IRT b",
            "Error",
            "Posición",
            "orden",
        ]
    ]
    est.columns = [
        "CodPregunta OCA",
        "Competencia",
        "Tema",
        "SubTema",
        "Categoria",
        "N",
        "Medición",
        "Error",
        "Posición\npregunta",
        "Orden Alternativas",
    ]
    archivo_est = BytesIO()
    est.to_excel(archivo_est)
    return archivo_est


def replace_equations(markup: str) -> Optional[str]:
    if pd.isna(markup):
        return
    soup = BeautifulSoup(markup, "lxml")
    imgs = soup.find_all(class_="Wirisformula")
    for img in imgs:
        mathml = img["data-mathml"].replace("«", "««").replace("»", "»»")  # type: ignore
        img.replace_with(mathml)
    # quitar <body></body> solo conservar lo de adentro
    result = str(soup.body)[6:-7]
    result = (
        result.replace("««", "<").replace("»»", ">").replace("¨", '"').replace("§", "&")
    )
    result = result.replace("<math ", '<math display="block"')
    matches = re.findall(
        r'<annotation encoding="LaTeX">.*?</annotation>', result, flags=re.DOTALL
    )
    for m in matches:
        result = result.replace(m, "")
    result = result.replace("<semantics>", "").replace("</semantics>", "")
    result = (
        result.replace("\xa0", " ")
        .replace(r"<p> </p>", " ")
        .replace(r'<p style="text-align: justify;"> </p>', " ")
    )
    return result


def fix_images(markup: str) -> Optional[str]:
    if pd.isna(markup):
        return
    soup = BeautifulSoup(markup, "lxml")
    imgs = soup.find_all("img")
    for img in imgs:
        if img["src"].startswith("/ftw/PR?"):  # type: ignore
            img["src"] = "https://app.fasttestweb.com" + img["src"]  # type: ignore
    return str(soup.body)[6:-7]


def process_items(df: pd.DataFrame) -> pd.DataFrame:
    # quitar algunos estilos
    df["Item Text"] = df["Item Text"].str.replace(
        "style=\"font-family: 'times new roman', times;\"", ""
    )
    df["Item Text"] = df["Item Text"].str.replace('style="font-size: 12pt;"', "")
    df["Item Text"] = df["Item Text"].str.replace(
        'style="text-align: justify; font-size: 12pt;"', 'style="text-align: justify;"'
    )

    # Reemplazar las acuaciones a MathML
    df["Item Text"] = df["Item Text"].apply(replace_equations)
    df["Answer 1"] = df["Answer 1"].apply(replace_equations)
    df["Answer 2"] = df["Answer 2"].apply(replace_equations)
    df["Answer 3"] = df["Answer 3"].apply(replace_equations)
    df["Answer 4"] = df["Answer 4"].apply(replace_equations)

    # Reemplazar las rutas de las imagenes
    df["Item Text"] = df["Item Text"].apply(fix_images)
    df["Answer 1"] = df["Answer 1"].apply(fix_images)
    df["Answer 2"] = df["Answer 2"].apply(fix_images)
    df["Answer 3"] = df["Answer 3"].apply(fix_images)
    df["Answer 4"] = df["Answer 4"].apply(fix_images)

    return df


def render_item(
    item_tpl: jinja2.Template, item: pd.Series, resaltar_clave: bool
) -> str:
    return item_tpl.render(
        description=item["Item Text"],
        # segun el vector generado cual debería ser la primera alternativa
        answer1=item[f"Answer {item['orden'][0]}"],
        answer2=item[f"Answer {item['orden'][1]}"],
        answer3=item[f"Answer {item['orden'][2]}"],
        answer4=item[f"Answer {item['orden'][3]}"],
        num_item=item["Ord"],
        padre=item["EsPadre"],
        num_texto=item["numtext"],
        salto=item["Salto"],
        cont=item["Continue"],
        blanca=item["Blanca"],
        resaltar_clave=resaltar_clave,
        clave=item["clave"],
    )


def generate(examen: Examen) -> BytesIO:
    jinja_env = jinja2.Environment(
        # donde están los templates, por defecto es la carpeta actual
        loader=jinja2.FileSystemLoader("templates"),
        autoescape=True,
    )

    df = load_files(examen)

    # generar archivo de estructura y archivo de claves
    examen.estructura = generate_estructura(df)
    examen.clave = generate_anskey(df, version=examen.version)  # type: ignore

    df = process_items(df)

    # Objetos para convertir un html a pdf usando chromium
    browser = get_browser()

    # Plantillas jinja
    item_tpl = jinja_env.get_template("item.tpl.html")
    prueba_tpl = jinja_env.get_template("test.tpl.html")
    background_tpl = jinja_env.get_template("background.tpl.html")

    # Assets
    with open("assets/styles.css", "r", encoding="utf-8") as f:
        styles = f.read()
    with open("assets/grid.svg", "r", encoding="utf-8") as f:
        grid = f.read()
    b64grid = base64.b64encode(grid.encode("utf-8")).decode("utf-8")

    # Generar el html de cada item
    df["html"] = df.apply(
        lambda x: render_item(item_tpl, x, examen.resaltar_clave), axis=1
    )

    start_item = 1
    start_page = 2
    for i, seccion in enumerate(examen.secciones):
        seccion.items_df = df.loc[df["Sec"] == seccion.nombre, :].copy()
        last = i == (len(examen.secciones) - 1)
        seccion.generate_sec_html(
            prueba_tpl, i + 1, styles, start_item, last, examen.extra_css
        )
        if seccion.derCuad:
            seccion.calculate_breaks(browser=browser, wait_for_id="finished")
            seccion.items_df["html"] = seccion.items_df.apply(
                lambda x: render_item(item_tpl, x, examen.resaltar_clave), axis=1
            )
            seccion.generate_sec_html(
                prueba_tpl, i + 1, styles, start_item, last, examen.extra_css
            )
        start_item = (
            start_item + seccion.items_df[~seccion.items_df["EsPadre"]].shape[0]
        )
        if seccion.html is not None:
            seccion.pdf = html2pdf(
                seccion.html, browser=browser, wait_for_id="finished"
            )

        seccion.generate_background_html(background_tpl, b64grid, i + 1, start_page)
        if seccion.bhtml is not None:
            seccion.bpdf = html2pdf(
                seccion.bhtml, browser=browser, wait_for_id="finished"
            )
        if seccion.pdf and seccion.bpdf:
            seccion.pdf_final = stamp_pdf(seccion.pdf, seccion.bpdf)
        start_page = start_page + seccion.num_pages()

    examen_pdfs = [
        seccion.pdf_final for seccion in examen.secciones if seccion.pdf_final
    ]
    if examen.caratula:
        examen_pdfs = [examen.caratula] + examen_pdfs
    examen.pdf = merge_pdf(examen_pdfs)
    examen.generar_configuracion_yaml()

    res = BytesIO()
    with ZipFile(res, "w") as z:
        z.writestr(
            f"PRUEBA-{examen.version}-{examen.codigo}.pdf", examen.pdf.getbuffer()
        )
        z.writestr(
            f"ESTRUCTURA-{examen.version}-{examen.codigo}.xlsx",
            examen.estructura.getbuffer(),
        )
        z.writestr(
            f"CLAVE-{examen.version}-{examen.codigo}.xlsx", examen.clave.getbuffer()
        )
        if examen.config_yaml is not None:
            z.writestr("config.yml", examen.config_yaml)
        if examen.caratula:
            z.writestr(f"CARÁTULA-{examen.version}.pdf", examen.caratula.getbuffer())
        for seccion in examen.secciones:
            if seccion.pdf_final:
                z.writestr(
                    f"{seccion.nombre}-{examen.version}-{examen.codigo}.pdf",
                    seccion.pdf_final.getbuffer(),
                )
            if examen.include_html and seccion.html:
                z.writestr(
                    f"{seccion.nombre}-{examen.version}-{examen.codigo}.html",
                    seccion.html,
                )
    return res


def load_yaml(obj: BytesIO) -> Examen:
    yml = yaml.safe_load(obj)
    return Examen.model_validate(yml)


def main():
    # Streamlit - para generar la "estructura" de la prueba

    st.set_page_config(initial_sidebar_state="collapsed")
    st.title("Diagramar prueba - FastTestWeb")

    with st.sidebar:
        est_config = st.file_uploader(
            "Archivo de configuración", help="El archivo de configuración"
        )
        include_html = st.checkbox(
            "Incluir los Html generados en el comprimido",
            value=False,
            help="Los archivos HTML sirven para depurar errores",
        )

    if est_config:
        default_examen = load_yaml(est_config)
    else:
        default_examen = Examen()

    datos = st.container()

    examen = {
        "version": datos.text_input(
            "Version",
            value=default_examen.version,
            help="Es para el nombre de archivo y asignar los temas y subtemas en la estructura (CIENCIAS,LETRAS,ARTE)",
        ),
        "codigo": datos.number_input(
            "Código",
            value=default_examen.codigo,
            format="%d",
            help="Dejar en 0 si se genera por primera vez, ingresar un código si se desea mantener siempre la mismas claves",
        ),
        "caratula": datos.file_uploader(
            "Carátula",
            help="Adjuntar una carátula si se desea incluir en la versión final",
        ),
        "resaltar_clave": datos.checkbox(
            "Resaltar clave",
            value=default_examen.resaltar_clave,
            help="Resalta la clave en amarillo para la revisión",
        ),
        "nsecciones": datos.number_input("Número de secciones", value=2, format="%d"),
        "secciones": [],
    }
    extra_styles = datos.checkbox(
        "Agregar estilos adicionales",
        value=default_examen.extra_css is not None,
        help="Marcar si se desea agregar estilos adicionales al contenido de la prueba",
    )

    if extra_styles:
        examen["extra_css"] = datos.text_area(
            "CSS extra",
            value=default_examen.extra_css,
            help="Agregar CSS extra al contenido de la prueba",
        )

    usar_password = datos.checkbox(
        "Agregar un password a los PDFs",
        value=default_examen.password is not None,
        help="Marcar si se desea agregar una contraseña a todos los archivos PDF",
    )

    if usar_password:
        examen["password"] = datos.text_input(
            "Password",
            value=default_examen.password,
            help="Password para los PDFs",
            type="password",
        )

    if examen["codigo"] == 0:
        examen["codigo"] = int(time.time())

    for i in range(examen["nsecciones"]):  # type: ignore
        if default_examen.nsecciones() != 0:
            default_seccion = default_examen.secciones[i]
        else:
            default_seccion = Seccion()
        container = st.container()
        container.subheader(f"Sección {i + 1}")
        st.write(default_seccion.derCuad)
        sec = {
            "archivo": container.file_uploader(
                f"Archivo {i + 1}",
                help="El archivo de metadata exportado por FastTestWeb",
            ),
            "nombre": container.text_input(
                f"Nombre {i + 1}",
                help="Esta etiqueta sale en la primera página de la sección y tambien en los bordes",
                value=default_seccion.nombre,
            ),
            "tiempo": container.text_input(
                f"Tiempo {i + 1}",
                value=default_seccion.tiempo,
                help="Esta etiqueta sale en la primera página",
            ),
            "saltos": container.text_input(
                f"Saltos de página {i + 1}",
                value=",".join(default_seccion.saltos),
                help="Indicar el número de ítem después del cual se quiere insertar un salto de página, separar por comas si se quiere indicar varios ej. 5,6,7",
            ),
            "derCuad": container.checkbox(
                f"La cara derecha es cuadriculada {i + 1}",
                value=default_seccion.derCuad,
                help="Si se desea que la cara derecha (abierto como libro) sea cuadriculada",
            ),
        }
        examen["secciones"].append(sec)
        examen["include_html"] = include_html

    submit = st.container()
    resultados = st.container()
    resultados.empty()

    btn = submit.button("PROCESAR")
    if btn:
        with resultados:
            with st.spinner("Generando archivos..."):
                ex = Examen.model_validate(examen)
                zip_file = generate(ex)
                st.header("Archivos generados")
                st.download_button(
                    "Descargar Archivos",
                    data=zip_file,
                    file_name=f"{ex.version}-{ex.codigo}.zip",
                    mime="application/zip",
                )


if __name__ == "__main__":
    main()
