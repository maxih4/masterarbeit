import http.client
import logging
import pathlib
import urllib.parse
from dataclasses import dataclass
import csv

import itertools
import json
from typing import Dict, List

from modules.rag.state import State
from utils.logging_config import configure_logging


@dataclass
class CsvInputObject:
    inhalt: str
    anonymisierung: str
    klassifizierung: str
    generierte_fragen: str
    wunschantwort: str


# get logging instance
def use_logger():
    configure_logging()
    return logging.getLogger(__name__)


# create http connection
def use_connection():
    return http.client.HTTPConnection("localhost", 8000)


# read files
def use_testset():
    rows: list[CsvInputObject] = []
    base = pathlib.Path(__file__).parent
    path = base / "files" / "input.csv"
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(
                CsvInputObject(
                    inhalt=row["Inhalt"].strip(),
                    anonymisierung=row["Anonymisierung"].strip(),
                    klassifizierung=row["Klassifizierung"].strip(),
                    generierte_fragen=row["Generierte Frage/Fragen"].strip(),
                    wunschantwort=row["Wunschantwort"].strip(),
                )
            )
    return rows


# for each loop

# make request to rag pipeline


counter = itertools.count(1001)


def request_and_response(
    connection: http.client.HTTPConnection, csv_object: "CsvInputObject"
) -> "State":  # returns State mapping
    thread_id = next(counter)  # e.g., itertools.count(1)
    params = {"sentence": csv_object.inhalt, "thread_id": thread_id}
    query = urllib.parse.urlencode(params)

    connection.request("GET", f"/search?{query}")  # note leading slash
    response = connection.getresponse()

    if response.status != 200:
        raise RuntimeError(f"Request failed with {response.status}: {response.reason}")

    body = response.read()  # wait until fully received
    data = json.loads(body.decode("utf-8"))
    return data


def __generate_headers__():
    header = [
        "anfrage",
        "wunsch_anonymisierung",
        "wunsch_klassifizierung",
        "wunsch_fragen",
        "wunschantwort",
    ]
    for x in range(8):
        header.append("generierte_frage_" + str(x))
        header.append("kontext_frage_" + str(x))
    header.append("anonymisierter_user_input")
    header.append("klassifizierung")
    steps = ["form_query", "generate"]
    for step in steps:
        header.append(step + "_token_usage")
    header.append("generierte_antwort")
    return header


def _to_cell(value) -> str:
    """Make sure every cell is a clean string."""
    if value is None:
        return ""
    # Keep str as-is
    if isinstance(value, str):
        return value
    # For lists/dicts/TypedDicts -> compact JSON (readable umlauts)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def __tokens_for_step__(state: State, step: str) -> str:
    """Return a string like 'input: X, output: Y' for the given step.
    Special case: step == 'generate' will also include 'generate_answer' and 'generate_summary'.
    """
    all_entries = state.get("token_usage", [])

    if step == "generate":
        step_names = {"generate", "generate_answer", "generate_summary"}
        entries = [e for e in all_entries if e.get("step_name") in step_names]
    else:
        entries = [e for e in all_entries if e.get("step_name") == step]

    if not entries:
        return "input: 0, output: 0"

    total_input = sum(e.get("input_tokens", 0) for e in entries)
    total_output = sum(e.get("output_tokens", 0) for e in entries)
    return f"input: {total_input}, output: {total_output}"


def __generate_row_element__(
    csvInputObject: CsvInputObject, state: State, headers: list[str]
):
    row = []
    for header in headers:
        match header:
            case "anfrage":
                val = csvInputObject.inhalt
            case "wunsch_anonymisierung":
                val = csvInputObject.anonymisierung
            case "wunsch_klassifizierung":
                val = csvInputObject.klassifizierung
            case "wunsch_fragen":
                val = csvInputObject.generierte_fragen
            case "wunschantwort":
                val = csvInputObject.wunschantwort

            case h if h.startswith("generierte_frage_"):
                idx = int(h.split("_")[-1])
                val = (
                    state["qa_pairs"][idx]["q"] if idx < len(state["qa_pairs"]) else ""
                )

            case h if h.startswith("kontext_frage_"):
                idx = int(h.split("_")[-1])
                val = (
                    state["qa_pairs"][idx]["ctx"]
                    if idx < len(state["qa_pairs"])
                    else ""
                )

            case "anonymisierter_user_input":
                val = state["user_input"]
            case "klassifizierung":
                val = state["classifier"]
            case "form_query_token_usage":
                val = __tokens_for_step__(state, "form_query")
            case "generate_token_usage":
                val = __tokens_for_step__(state, "generate")
            case "generierte_antwort":
                val = state["answer"]
            case _:
                val = ""
        row.append(val)
    return row


def write_csv(
    data: list[tuple[CsvInputObject, State]], filename: str = "output.csv"
) -> None:
    header = __generate_headers__()
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for csvInputObject, state in data:
            row = __generate_row_element__(csvInputObject, state, header)
            writer.writerow(row)


def main():
    logger = use_logger()
    connection = use_connection()
    testset = use_testset()

    results: list[tuple[CsvInputObject, State]] = []
    for row in testset:
        logger.info(row.inhalt)
        response = request_and_response(connection, row)
        results.append((row, response))
        # remove break to process all

    if results:
        write_csv(results, filename="evaluation_results.csv")
        logger.info("CSV export completed: evaluation_results.csv")
    else:
        logger.warning("No results to write to CSV.")


if __name__ == "__main__":
    main()
