import argparse
from collections import defaultdict
import gc
import json
import os
import re
import sys
import time
import pandas as pd
from tabulate import tabulate
import torch
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = """Eres un analizador riguroso de bases de datos relacionales y código SQL.
Tu tarea es clasificar la consulta SQL provista con respecto al esquema DDL y el requerimiento textual en lenguaje natural.

Debes elegir ESTRICTAMENTE una de las siguientes cuatro categorías:
1. "syntactic": Errores de gramática SQL, palabras clave mal escritas o sintaxis inválida.
2. "semantic": Errores referenciales contra el esquema DDL (tablas/columnas inexistentes o tipos de datos incompatibles).
3. "conceptual": La sintaxis y referencias al esquema son válidas, pero la lógica relacional o condiciones no resuelven la necesidad del requerimiento.
4. "hallucination": La consulta es completamente correcta y resuelve el requerimiento según el DDL (no contiene errores).

Responde ÚNICAMENTE un objeto JSON válido con la siguiente estructura:
{"classification": "<syntactic|semantic|conceptual|hallucination>"}"""

CATEGORIES = ["syntactic", "semantic", "conceptual", "hallucination"]

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SQL local con vLLM y liberación de VRAM.")
    parser.add_argument("--datos", type=str, default="datos.json", help="Ruta al archivo JSON ya curado con los casos.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.6-35B-A3B", help="Identificador del modelo en Hugging Face o ruta local.")
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv", help="Archivo CSV de salida.")
    parser.add_argument("--gpu-mem", type=float, default=0.90, help="Fracción de VRAM a utilizar.")
    parser.add_argument("--max-len", type=int, default=4096, help="Longitud máxima de contexto.")
    return parser.parse_args()

def load_curated_dataset(ruta):
    """Carga directa del archivo JSON curado."""
    if not os.path.exists(ruta):
        print(f"[ERROR] No se encontró el archivo: {ruta}")
        sys.exit(1)
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Dataset curado cargado exitosamente: {len(data)} casos listos para evaluar.")
    return data

def parse_prediction(raw_text):
    """Extrae la clasificación asegurando concordancia estricta con las 4 categorías."""
    try:
        data = json.loads(raw_text)
        pred = data.get("classification", "").lower().strip()
        if pred in CATEGORIES:
            return pred
    except Exception:
        pass

    match = re.search(r'"classification"\s*:\s*"([a-zA-Z]+)"', raw_text)
    if match:
        pred = match.group(1).lower().strip()
        if pred in CATEGORIES:
            return pred

    return "unknown"

def run_benchmark(llm, dataset):
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        response_format={"type": "json_object"}
    )

    results = []
    print("\n[INFO] Ejecutando inferencia sobre el dataset curado...\n")

    for idx, case in enumerate(dataset, start=1):
        case_id = case.get("id", idx)
        expected = case["category"].lower().strip()
        ddl = case.get("ddl", "")
        nl_query = case.get("nl_query", "")
        sql_query = case.get("sql_query", "")

        user_content = (
            f"ESQUEMA DDL:\n{ddl}\n\n"
            f"REQUERIMIENTO EN LENGUAJE NATURAL:\n{nl_query}\n\n"
            f"CONSULTA SQL A EVALUAR:\n{sql_query}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        start_time = time.time()
        outputs = llm.chat(messages=messages, sampling_params=sampling_params, use_tqdm=False)
        elapsed = time.time() - start_time

        raw_output = outputs[0].outputs[0].text
        predicted = parse_prediction(raw_output)
        is_correct = (predicted == expected)
        status = "OK" if is_correct else "ERR"

        results.append({
            "id": case_id,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "latency_sec": round(elapsed, 3),
            "sql_query": sql_query
        })

        print(f"[{status}] Caso {case_id:03d} | Esperado: {expected:<13} | Predicho: {predicted:<13} | Tiempo: {elapsed:.2f}s")

    return results

def print_metrics(results, output_csv):
    df = pd.DataFrame(results)

    # 1. Exactitud por categoría (según Tabla 1 del paper)
    summary_data = []
    for cat in CATEGORIES:
        sub_df = df[df["expected"] == cat]
        total_cat = len(sub_df)
        hits = sub_df["correct"].sum() if total_cat > 0 else 0
        acc = (hits / total_cat * 100) if total_cat > 0 else 0.0
        summary_data.append({
            "Categoría": cat,
            "Total": total_cat,
            "Aciertos": hits,
            "Exactitud (%)": f"{acc:.2f}%"
        })

    total_cases = len(df)
    total_hits = df["correct"].sum()
    global_acc = (total_hits / total_cases * 100) if total_cases > 0 else 0.0
    mean_latency = df["latency_sec"].mean()

    print("\n" + "=" * 55)
    print("RESUMEN DE RESULTADOS DEL BENCHMARK")
    print("=" * 55)
    print(tabulate(summary_data, headers="keys", tablefmt="github"))
    print(f"\nExactitud Global : {global_acc:.2f}% ({total_hits}/{total_cases})")
    print(f"Latencia Media   : {mean_latency:.3f} s por inferencia")

    # 2. Matriz de Confusión
    confusion = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        confusion[row["expected"]][row["predicted"]] += 1

    matrix_rows = []
    for exp in CATEGORIES:
        row_dict = {"Esperado / Predicho": exp}
        for pred in CATEGORIES:
            row_dict[pred] = confusion[exp][pred]
        matrix_rows.append(row_dict)

    print("\nMATRIZ DE CONFUSIÓN:")
    print(tabulate(matrix_rows, headers="keys", tablefmt="grid"))

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[INFO] Detalle completo exportado en: {output_csv}")

def release_memory(llm):
    """Destruye el objeto del modelo y vacía la VRAM de la GPU por completo."""
    print("\n[INFO] Liberando recursos y vaciando VRAM...")
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[INFO] VRAM liberada satisfactoriamente.")

def main():
    args = parse_args()
    dataset = load_curated_dataset(args.datos)

    print(f"\n[INFO] Cargando modelo '{args.model}' en la GPU con vLLM...")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_len,
        trust_remote_code=True
    )

    try:
        results = run_benchmark(llm, dataset)
        print_metrics(results, args.output_csv)
    finally:
        # Asegura la destrucción y liberación de memoria incluso si ocurre un error
        release_memory(llm)

    print("[INFO] Ejecución concluida. La memoria de la GPU ha vuelto a su estado original.\n")

if __name__ == "__main__":
    main()