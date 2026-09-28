# Velo Demand — Render optimized

## Что положить в `data/`

Нужны:
- `h3_hexagons_12_600_street_tr.parquet`
- `h3_cl_24_07.parquet`
- `h3_cl_25_07.parquet`
- `h3_cl_26_07.parquet`

`prepare_h3.py` один раз при build создаёт:
- `data/h3_lookup.parquet`

Это GeoParquet вместо WKT, поэтому приложение не парсит 600k WKT при каждом пользовательском расчёте.

## Оптимизация

1. H3-слой загружается один раз и кэшируется в процессе Python.
2. Перед spatial join берётся bbox 5-метровых буферов загруженных велодорог.
3. Spatial join выполняется только по H3-кандидатам внутри этого bbox.
4. Один Render worker специально оставлен, чтобы не дублировать 600k H3-полигонов в RAM.
5. Расчёт идёт в background thread, UI показывает прогресс.
6. GeoJSON/CSV результата лежат в job-папке и доступны для скачивания.

## Render

Подключите GitHub-репозиторий с этими файлами. `render.yaml` сам задаёт build/start commands.

Важно: файловая система Render Web Service эфемерная. Результаты предназначены для текущей сессии; для постоянного хранения нужен object storage.

## Локальный запуск

```bash
pip install -r requirements.txt
python prepare_h3.py
python app.py
```

Откройте http://localhost:5000
