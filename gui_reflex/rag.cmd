@echo off
echo Avvio Financial RAG UI...

REM Log fuori dalla cartella progetto Reflex
set RAG_LOG_DIR=C:\AI_RAG_LOGS
set AUDIT_LOG_PATH=C:\AI_RAG_LOGS\rag_audit.jsonl
set EVAL_LOG_PATH=C:\AI_RAG_LOGS\rag_eval_log.jsonl

REM Disattiva evaluation automatica dentro la UI
set EVAL_ENABLED=0

REM UI più leggera
set MAX_UI_SOURCES=8
set MAX_UI_SOURCE_CONTENT_CHARS=900
set MAX_UI_DEBUG_CHARS=6000

reflex run
pause