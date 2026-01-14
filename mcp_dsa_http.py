from fastmcp import FastMCP, Context
from base64 import b64encode
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Any, ClassVar
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from google.cloud import language_v2
from google.oauth2 import service_account
import nltk
from nltk.corpus import stopwords
import spacy
import pandas as pd
import re
import aiohttp
import os


# ===========================
# Cargar variables de entorno
# ===========================

load_dotenv()

USER=os.getenv('USERNAME')
PASS=os.getenv('PASSWORD')
GNLP=os.getenv('GOOGLE_NLP')

# ===========================
# Create an MCP server
# ===========================

mcp = FastMCP(name='Keyword Research', instructions='''Este servidor es para hacer consultas de keywords relacionadas para SEO.''')

# ===========================
# Clase para conectarse a la Api de DFS
# ===========================

class KWResearch(BaseModel):
    username: str
    password: str
    domain: ClassVar[str] = "api.dataforseo.com"

    async def request(self, path, method, data=None):
        auth = b64encode(f"{self.username}:{self.password}".encode()).decode()
        headers = {'Authorization': f'Basic {auth}'}
        
        async with aiohttp.ClientSession() as session:
            url = f"https://{self.domain}{path}"
            if method == 'GET':
                async with session.get(url, headers=headers) as response:
                    return await response.json()
            elif method == 'POST':
                async with session.post(url, headers=headers, json=data) as response:
                    return await response.json()

    async def get(self, path):
        return await self.request(path, 'GET')
    
    async def post(self, path, data):
        return await self.request(path, 'POST', data)
    
cliente = KWResearch(username=USER, password=PASS)

# ===========================
# Tools para el LLM
# ===========================

@mcp.tool()
async def KeywordsSugeridas(keyword:str,
                            locacion_codigo:int = None):

    '''
    Description:
        Tool para extraer keywords sugeridas de la base de datos LABs de Data For SEO.
        Puedes devolver: Keywords, volumen de búsqueda, keyword difficulty, main domain rank.
        Tienes el recurso Locaciones si no tienes el códido de locacion.

    Args:
        keyword: Keyword o query que usuario quiere buscar.
        locacion codigo: codigo de geolocalización de los resultados.
    
    Return:
        keyword encontrada, volumen de búsqueda, dificultad palabra clave encontrada,
        info general backlinks, info general domain DR. Sí o sí debes devolver esto.
        Debes decir la cantidad de resultados extraídos.
    '''

    if not USER or not PASS:
       return {'respuesta_llm': "Faltan las variables de entorno USERNAME o PASSWORD"}

    post_data = dict()            
    post_data[len(post_data)] = dict(
        keyword = keyword,
        location_code = locacion_codigo,
        include_serp_info = False,
        include_seed_keyword = False)
    response = await cliente.post("/v3/dataforseo_labs/google/keyword_suggestions/live", post_data)
    lista = []
    if response['tasks'][0]['result'][0]['items']:
        try:
            for i in response['tasks'][0]['result'][0]['items']:
                keyword_encontrada = i['keyword']
                if not keyword_encontrada:
                    keyword_encontrada = 'sin_keyword'
                volumen = i['keyword_info']['search_volume']
                if not volumen:
                    volumen = 0
                dificultad = i['keyword_properties']['keyword_difficulty']
                if not dificultad:
                    dificultad = 0
                try:
                    bl_avg = i['avg_backlinks_info']['rank']
                    main_domain_rank = i['avg_backlinks_info']['main_domain_rank']
                except:
                    bl_avg = 0
                    main_domain_rank = 0
                lista.append({'keyword encontrada':keyword_encontrada,
                               'volumen de búsqueda': volumen, 
                               'dificultad palabra clave encontrada': dificultad, 
                               'info general backlinks':bl_avg,
                               'info general domain DR':main_domain_rank})
            if lista:
                return {'keywords':lista}
            else:
                await ctx.error('No hay resultados para mostrar')
        except Exception as e:
            return {'respuesta_llm':[e]}
    
@mcp.tool()
async def KeywordsParaURLsYDominios(ctx: Context,
                                    urls: list[str],
                                    limite: int = 1000,
                                    locacion_codigo: int = None,
                                    filtro_trafico: int = None,
                                    filtro_kw: list[str] = None):

    '''
    Description:
        Tool para ver keywords sugeridas de paginas específicas o dominios.
        Debes identificar si el usuario quiere que sea una URL completa o un dominio. No cambies lo que el usuario necesita.
        Debes hacer la petición por 1000 keywords, no menos de eso, pero debes decirle, no preguntarle.
        Preguntale siempre al usuario cuando de trafico minimo deben tener las keywords.
    
    args:
        - Urls para analizar, ejemplo: dominio.com, https://www.dominio.com/ruta, https://www.dominio.com, https://dominio.com.
        Usa siempre lo que el usuario necesita y no lo que tú quieres. Pregunta en caso de necesitarlo.
        - Identifica si es dominio o si es una URL, ya que son objetivos de analisis diferentes.
        - locacion codigo, limite de resultados por busqueda, filtro_trafico es el minimo de trafico que deben tener las keywords.
        - Tienes el recurso Locaciones si no tienes el códido de locacion.
    
    return:
        str: Texto en una línea por resultado con el formato:
        'URL buscada: {url} | Keyword: {keyword} | Volumen: {search_volume} | Competencia: {competition} |
        ETV: {impresions_etv} | URL encontrada: {url_encontrada} | DR: {page_rank} | Domain DR: {main_domain_rank} |
        Posición SERP: {position} | Título SERP: {title}'
        Debes responder decir la cantidad de resultados extraídos.
    '''

    if not USER or not PASS:
        await ctx.error("Faltan las variables de entorno USERNAME o PASSWORD")


    data_total = []
    for url in urls:
        if 'https://www.' in url:
            repl = url.replace('https://www.', '')
        elif 'https://' in url:
            repl = url.replace('https://', '')
        else:
            repl = url

        dominio = repl.split('/')[0]
        try:
            path = re.split(r'(/.*)', repl)[1]
            filters = [["ranked_serp_element.serp_item.relative_url", "like", f'%{path}%']]
        except:
            filters = []

        if filtro_trafico is not None:
            if filters:
                filters.extend(["and", ["ranked_serp_element.serp_item.etv", ">", filtro_trafico]])
            else:
                filters.extend([["ranked_serp_element.serp_item.etv", ">", filtro_trafico]])


        if filtro_kw is not None:
            filtro_kw = '|'.join(filtro_kw)
            if filters:
                filters.extend(["and", ["keyword_data.keyword", "regex", filtro_kw]])
            else:
                filters.extend([["keyword_data.keyword", "regex", filtro_kw]])
        
        post_data = dict()
        post_data[len(post_data)] = dict(
            target=dominio,
            location_code=locacion_codigo,
            filters=filters,
            limit=limite)
        
        await ctx.debug(f"Extrayendo data de Data for SEO")
        response = await cliente.post("/v3/dataforseo_labs/google/ranked_keywords/live", post_data)
        try:
            if response['tasks'][0]['result'][0]['items']:
                for i in response['tasks'][0]['result'][0]['items']:
                    keyword = i['keyword_data']['keyword']
                    search_volume = i['keyword_data']['keyword_info']['search_volume'] or 0
                    competition = i['keyword_data']['keyword_properties']['keyword_difficulty'] or 'nada'
                    impresions_etv = i['ranked_serp_element']['serp_item']['etv'] or 0
                    url_encontrada = i['ranked_serp_element']['serp_item']['url'] or 'nada'
                    page_rank = i['ranked_serp_element']['serp_item']['rank_info']['page_rank'] or 0
                    main_domain_rank = i['ranked_serp_element']['serp_item']['rank_info']['main_domain_rank'] or 0
                    position = i['ranked_serp_element']['serp_item']['rank_absolute'] or 0
                    title = i['ranked_serp_element']['serp_item']['title'] or 'nada'
                    data_total.append({
                        'url_buscada': url, 
                        'keyword': keyword, 
                        'volumen': search_volume, 
                        'competencia': competition, 
                        'etv': impresions_etv,
                        'url':url_encontrada,
                        'DR':page_rank,
                        'domain_DR':main_domain_rank,
                        'posicion_serp':position,
                        'title_serp':title})
            else:
                await ctx.error(f'No hay datos para: {repl}')
        except Exception as e:
            await ctx.error(f'Pasó esto con {repl}: {str(e)}')
    if data_total:
        return {'respuesta llm': data_total}
    else:
        await ctx.error('No hay resultados para mostrar')
    
@mcp.tool()
async def SerpCompetidores(ctx: Context,
                     keywords:List[str],
                     locacion_codigo:int,
                     cant_serp:int = 10):

    '''
    Description:
        Tool para extraer resultados de Google, luego keywords que posicionan esas páginas.
        Antes de ejecutar debes preguntar siempre la locación donde el usuario quiere hacer la consulta.

    Args:
        Keywords: listado de keywords que queremos analizar la serp
        Locacion codigo: codigo de pais que el usuario quiere geolocalizar la búsqueda
        Cantidad SERP: Cantidad de resultados de Google a extraer por cada keyword

    Return:
        Devuelve un diccionario con la respuesta del llm, que es una lista con todos los resultados. Dile al usuario que la data esta
        lista en caso que la haya.
    '''

    if not USER or not PASS:
        await ctx.error("Faltan las variables de entorno USERNAME o PASSWORD")

    resultados_serp = []
    for kw in keywords:
        post_data = dict()
        post_data[len(post_data)] = dict(
            location_code = locacion_codigo,
            language_code ='es',
            keyword = kw,
            device = 'desktop',
            depth = cant_serp)
        
        response = await cliente.post("/v3/serp/google/organic/live/regular", post_data)
        try:
            for i in response['tasks'][0]['result'][0]['items']:
                url = i['url']
                rank = i['rank_absolute']
                title = i['title']
                resultados_serp.append({'keyword_buscada':kw, 'url_encontrada':url, 'ranking_organico':rank, 'title_serp':title})
        except Exception as e:
            await ctx.error(f'Hubo el siguiente problema: {e}')
    if resultados_serp:
        return {'respuesta llm': resultados_serp[:3]}
    else:
        await ctx.error('No hay resultados para mostrar')


# -> Dict[str, List[Dict[str, Any]]]
@mcp.tool()
async def KeywordsRelacionadas(ctx:Context,keyword: str,
                          nivel: int = 4,
                          limite: int = 1000,
                          locacion_codigo: int = None):
    '''
    Busca palabras clave relacionadas junto con sus métricas de volumen de búsqueda y competencia.
    Esta función asíncrona consulta la API de DataForSeo para encontrar palabras clave relacionadas 
    con una palabra clave semilla, incluyendo sus volúmenes de búsqueda y niveles de competencia.

    Parámetros:
        ctx (Context): El objeto de contexto para la función asíncrona
        keyword (str): La palabra clave semilla para encontrar términos relacionados
        nivel (int): El nivel de profundidad para la búsqueda de palabras clave relacionadas. Por defecto es 4 y obligatorio
        limite (int, opcional): Número máximo de resultados a devolver. Por defecto es 1000 y obligatorio.
        locacion_codigo (int, opcional): Código de ubicación para segmentación geográfica. Por defecto None

    Return:
        Devuelve un diccionario con la respuesta del llm, que es una lista con todos los resultados. Dile al usuario que la data esta
        lista en caso que la haya.
    '''

    if not USER or not PASS:
        await ctx.error("Faltan las variables de entorno USERNAME o PASSWORD")

    lista_final = []
    post_data = dict()
    post_data[len(post_data)] = dict(keyword=keyword,
                                        location_code= locacion_codigo,
                                        language_code='es',
                                        depth= nivel, 
                                        filters=[["keyword_data.keyword_info.search_volume", ">", 10]],
                                        limit= limite)
    response = await cliente.post("/v3/dataforseo_labs/google/related_keywords/live", post_data)
    try:
        for i in response['tasks'][0]['result'][0]['items']:
            keyword = i['keyword_data']['keyword']
            volumen = i['keyword_data']['keyword_info']['search_volume']
            if not volumen:
                volumen = 0
            competencia = i['keyword_data']['keyword_info']['competition']
            print(competencia)
            if not competencia:
                competencia = 0
            lista_final.append({'keyword_encontrada': keyword, 'volumen':volumen, 'competencia':int(competencia*100)})
    except Exception as e:
        await ctx.error(f'Hubo el siguiente problema: {e}')
    if lista_final:
        return {'respuesta llm': lista_final}
    else:
        await ctx.error('No hay resultados para mostrar')


@mcp.tool()
async def TopicalAuthority(ctx: Context,
                           keywords:List[str],
                           locacion_codigo:int):
        
        '''
        Tool para sacar la autoridad de tópico. Debes consultar siempre al usuario si tiene un listado de keywords.
        Si no lo tiene, debes utilizar sí o sí de forma obligatoria, la tool KeywordsSuggestions antes con una keyword específica
        que el usuario debe proporcionarte.
        Debes tener el listado de keywords antes de hacer el análisis de tópico, sino, no inventes keywords para usar la tool.

        Args:
            Keywords: Listado de keywords que vas a analizar para tener la autoridad de tópico. Debes preguntar al usuario o utilizar KeywordsSuggestions,
            como te dije anteriormente.

        Return:
            Debes devolver una lista con diccionarios con los resultados. Debes ordenar por visibilidad descendente y posiciones ascendente, junto
            con la cantidad de keywords por competidore que fueron encontradas.
        '''

        if not USER or not PASS:
            await ctx.error("Faltan las variables de entorno USERNAME o PASSWORD")

        post_data = dict()
        post_data[len(post_data)] = dict(
            keywords=keywords,
            location_code=locacion_codigo,
            language_name='Spanish')

        response = await cliente.post("/v3/dataforseo_labs/google/serp_competitors/live", post_data)
        dominios = []
        try:
            for a in response['tasks'][0]['result'][0]['items']:
                dominio = a['domain'].replace('www.', '') if 'www' in a['domain'] else a['domain']
                cant_kws = a['keywords_count']
                vis = a['visibility']
                pos = a['median_position']
                dominios.append({'dominio':dominio, 'visibilidad':vis, 'posiciones': pos, 'cantidad_keywords':cant_kws})
        except Exception as e:
            ctx.error(f'Ocurrió lo siguiente: {e}')
        if dominios:
            return dominios
        else:
            return 'No hay resultados'


        
async def crawl_sequential(urls: List[str]):
    browser_config = BrowserConfig(
        headless=True,
        # For better performance in Docker or low-memory environments:
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"])
    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator())
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    try:
        session_id = "session1"
        markdowns = [] 
        for url in urls:
            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id=session_id)
            if result.success:
                markdown = result.markdown.raw_markdown
                data_mark = {'url':url,
                             'markdown':str(markdown)} 
                markdowns.append(data_mark)
    finally:
        await crawler.close()
    return markdowns


@mcp.tool()
async def CrawlerURLs(ctx: Context,
                      urls: List[str]):
    '''
        Description:
            Tool escrapea información de los sitios que el usuario necesite.
        
        Args:
            urls: lista de URLs que el usuario necesita escrapear su información.
            
        Return:
            Respuesta llm: Una lista con diccionarios donde está la información extraída de los sitios en fomato markdowns.
        '''
    
    ctx.info("Crawleando URLs")
    data_crawler = await crawl_sequential(urls=urls)
    return {'respuesta llm': data_crawler}



nltk.download('stopwords')
def clean_text(text):
    print('Limpiando textos')
    nlp = spacy.load('es_core_news_sm')
    stop_words = set(stopwords.words('english') + stopwords.words('spanish'))
    doc = nlp(text)
    cleaned_tokens = [
        token.orth_ for token in doc 
        if token.text.lower() not in stop_words 
        and token.is_alpha 
        and not token.like_num]
    return ' '.join(cleaned_tokens)


@mcp.tool()
async def AnalizarEntidades(ctx: Context, textos: str):

    '''
    Description:
        Tool que hace una busqueda de entidades en la API NLP de Google.
    
    Args:
        Textos: String que debe entregar sí o sí el usuario. Debes pedirlo siempre.
        ruta_credenciales: json de las credenciales de Google.
        
    Return:
        Respuesta llm: Una lista con diccionarios donde están las entidades. Di que todo fue enxtraido con éxito.
        Debes responder con un resumen de las entidades con mayor probailidad (es_o_no_es), formato tabla, decir la cantidad
        de resultados extraídos.
        Haz un resumen de cómo conectan semánticamente las entidades extraídas.
    '''

    if not GNLP:
        ctx.error(f'Faltan las credenciales. Sin esto no puedes continuar. Pidelas al usuario.')

    texto_limpio = clean_text(textos)
    ctx.debug('Analizando textos')
    creds = service_account.Credentials.from_service_account_file(GNLP)
    client = language_v2.LanguageServiceClient(credentials=creds)
    data_nlp = []
    categories_dict = {}
    document = {
        "content": texto_limpio,
        "type_": language_v2.Document.Type.PLAIN_TEXT,
        "language_code": "es"}
    features = {
        "extract_entities": True,
        "extract_document_sentiment": False,
        "classify_text": True,
        "moderate_text": False}
    response = client.annotate_text(
        request={
            "document": document,
            "features": features,
            "encoding_type": language_v2.EncodingType.UTF8})
    for category in response.categories:
        cat_name = category.name
        cat_conf = f'{category.confidence:.2%}'
        categories_dict.update({cat_name: cat_conf})
    for entity in response.entities:
        entity_dict = {
            'nombre_ent': entity.name,
            'tipo_ent': language_v2.Entity.Type(entity.type_).name,
            'categories': categories_dict}
        for mention in entity.mentions:
            mencion = language_v2.EntityMention.Type(mention.type_).name
            es_o_no_es = mention.probability
            entity_dict.update({
                'mencion': mencion,
                'es_o_no_es': es_o_no_es})
            data_nlp.append(entity_dict)
    return {'respuesta llm': data_nlp}

    
@mcp.tool()
async def Locaciones(ctx: Context, pais: str):

    '''
    Description:
        Tool para proporcionar códigos para otras herramientas cuando el usuario pide keywords o dominios para un
        país en específico. Debes siempre consultar al usuario que locación quiere antes de ejecutar la herramienta.

    Args:
        pais: País que pide el usuario un código de locacion.

    Return:
        Código de locación f
    '''

    data = {
        'españa': 2724,
        'argentina': 2032,
        'chile': 2152,
        'ecuador': 2862,
        'bolivia': 2068,
        'venezuela': 2218,
        'brasil': 2076,
        'uruguay': 2218}
    
    format_pais = pais.lower().strip()

    try:
        data[format_pais]
        return {'pais':data[format_pais]}
    except Exception as e:
        await ctx.error(f'Error pais buscado {e}')
        return {'respuesta llm': f'Entrega un país que disponible como {data.keys()}'}


# ===========================
# Main de transport
# ===========================

async def main():
    await mcp.run_async(
        transport="http",
    )

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
    


