# Learn Basic RAG

Un RAG local y pequeño para **leer el código, ejecutarlo y entender el recorrido de una pregunta**. Usa LangChain, Chroma, Ollama y Streamlit. Los documentos son ficticios: no contiene datos de una empresa ni requiere una API de pago.

## Qué ocurre

```text
INGESTA (al preparar o actualizar documentos)
Markdown → Document(texto + fuente) → chunks → EmbeddingGemma → Chroma en disco

CONSULTA (por cada pregunta)
Pregunta → EmbeddingGemma → búsqueda en Chroma → fragmentos + pregunta → Qwen → respuesta
```

**RAG no entrena el modelo.** Añade evidencia al contexto de cada consulta. EmbeddingGemma representa textos; Qwen redacta. La búsqueda devuelve candidatos, no una garantía de que exista la respuesta.

## Ejecutarlo

Necesitas [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.14.7 y [Ollama](https://ollama.com/download). La primera preparación descarga dependencias y modelos; puede tardar y requiere espacio en disco. La inferencia ocurre en tu equipo.

Abre una terminal en la raíz del repositorio:

```powershell
uv sync --locked
ollama pull embeddinggemma:latest
ollama pull qwen3:4b
ollama list
uv run python -m scripts.build_index
uv run streamlit run ui/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Si ya tienes los modelos, omite los `pull`. Si Ollama no responde, abre su aplicación o ejecuta `ollama serve` en otra terminal; no inicies otra instancia si ya está activo. Los comandos fijan los nombres de modelo, pero `latest` es una etiqueta mutable: `ollama list` permite consultar el ID instalado.

También puedes usar solo la terminal:

```powershell
uv run python -m scripts.search
uv run python -m scripts.ask
```

`search` muestra evidencia; `ask` genera una respuesta. No reconstruyas el índice por cada pregunta.

## Ruta de lectura: menos de una hora como objetivo

La duración depende de tus conocimientos de Python y no incluye instalaciones. Sigue este orden; `examples/` es opcional:

| Archivo | Qué entender |
|---|---|
| `src/learn_basic_rag/ingestion.py` | Lee Markdown y divide el texto conservando `source`. |
| `src/learn_basic_rag/embeddings.py` | Hereda la conexión de LangChain y añade los prefijos de EmbeddingGemma. |
| `scripts/build_index.py` | Genera y almacena vectores, texto y metadatos. |
| `src/learn_basic_rag/retrieval.py` | Representa la pregunta y recupera candidatos. |
| `src/learn_basic_rag/generation.py` | Construye los mensajes con evidencia y llama a Qwen. |
| `scripts/ask.py` y `ui/app.py` | Conectan las mismas funciones desde terminal o chat. |

Tres ideas para revisar en el código:

- `Document` guarda texto y metadatos: todavía no es un vector.
- `chunk_size=500` y `chunk_overlap=80` se miden en **caracteres**, no tokens. Son valores didácticos, no parámetros optimizados. EmbeddingGemma admite entradas de hasta 2048 tokens; el prefijo también cuenta.
- `k=1` pide el candidato más cercano aunque sea irrelevante. El prompt pide abstención, pero no la garantiza. La lista vacía se controla con código; la evidencia insuficiente todavía depende del modelo.

En Streamlit, el historial es **visual y temporal**: no se envía a Qwen. Limpiar conversación no elimina el índice.

## Comprobar y experimentar

```powershell
uv run python -m pytest -q
```

Los tests rápidos comprueban contratos; los de generación simulan el modelo y **no miden su calidad real**. Los avisos de obsolescencia de dependencias no equivalen a fallos del proyecto.

Prueba manualmente:

| Pregunta | Comportamiento esperado, no garantizado |
|---|---|
| ¿La garantía cubre daños por golpes? | No; evidencia de `garantias.md`. |
| ¿Atienden los domingos? | No; evidencia de `horarios.md`. |
| ¿Cuánto cuesta el servicio? | Reconocer que no hay información de precios. |

Abre los fragmentos recuperados y compara cada afirmación con su evidencia. Una fuente mostrada no implica una cita verificada automáticamente.

**Reto:** añade una política ficticia de precios en `data/documents/`, reconstruye y repite la última pregunta:

```powershell
uv run python -m scripts.build_index --reset
```

`--reset` elimina y reconstruye **solo** `learning_documents`. Úsalo cuando cambies o elimines archivos para no conservar chunks obsoletos. No es una actualización incremental ni transaccional: si falla la ingesta después del borrado, repítela. Reinicia la app si la tienes abierta durante la reconstrucción.

## Límites y autoría

Demo educativa de flujo fijo, **no un agente ni un sistema productivo**. No incluye usuarios/permisos, OCR, PDF, reranker, verificación automática de citas, evaluación sistemática, sincronización incremental ni control exacto de tokens. No la expongas a internet ni uses datos confidenciales. El prompt no elimina alucinaciones ni garantiza defensa contra instrucciones maliciosas en documentos.

Desarrollado como práctica guiada con asistencia de IA. `.venv/`, `.chroma/` y secretos locales no deben publicarse; sí `pyproject.toml`, `uv.lock`, código y documentos ficticios. El código de este repositorio se distribuye bajo la [licencia MIT](LICENSE). Las dependencias y modelos conservan sus respectivas licencias.
