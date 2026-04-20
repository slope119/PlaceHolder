"""
Script para plotar a comparação entre múltiplas execuções de treinamento federado.

# Plota todos os CSVs da pasta results/ automaticamente
python plot_comparison.py

# Ou escolhe arquivos específicos
python plot_comparison.py results/alie_*.csv results/benign_*.csv

# Muda o nome do arquivo de saída
python plot_comparison.py --output meu_grafico.png

"""


import argparse
import csv
import os
import sys
import matplotlib.pyplot as plt


def load_csv(filepath: str) -> dict:
    """Carrega um CSV de resultados e retorna um dicionário com rounds, accuracy e loss."""
    rounds, accuracy, loss = [], [], []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            accuracy.append(float(row["accuracy"]))
            loss.append(float(row["loss"]))
    return {"rounds": rounds, "accuracy": accuracy, "loss": loss}


def label_from_filename(filepath: str) -> str:
    """Gera um label legível a partir do nome do arquivo CSV.

    Exemplo: results/alie_frac0.4_20260420_153000.csv → alie (frac=0.4)
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    parts = basename.split("_")

    # Formato esperado: <attack>_frac<fraction>_<timestamp>
    attack = parts[0]
    frac = ""
    for part in parts[1:]:
        if part.startswith("frac"):
            frac = part.replace("frac", "")
            break

    if frac:
        return f"{attack} (frac={frac})"
    return attack


def plot_comparison(filepaths: list[str], output: str) -> None:
    """Plota accuracy e loss de múltiplas execuções no mesmo gráfico."""
    if not filepaths:
        print("Nenhum arquivo CSV encontrado.")
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparação entre Execuções — Aprendizado Federado")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, filepath in enumerate(sorted(filepaths)):
        data = load_csv(filepath)
        label = label_from_filename(filepath)
        color = colors[i % len(colors)]

        ax1.plot(data["rounds"], data["accuracy"], marker="o", label=label, color=color)
        ax2.plot(data["rounds"], data["loss"], marker="o", label=label, color=color)

    ax1.set_title("Acurácia Global")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Loss Global")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Gráfico de comparação salvo em {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plota comparação entre execuções federadas.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Arquivos CSV a incluir no gráfico. Se omitido, usa todos em results/",
    )
    parser.add_argument(
        "--output",
        default="comparison.png",
        help="Nome do arquivo de saída (padrão: comparison.png)",
    )
    args = parser.parse_args()

    if args.files:
        filepaths = args.files
    else:
        results_dir = "results"
        if not os.path.isdir(results_dir):
            print(f"Pasta '{results_dir}' não encontrada. Execute ao menos uma simulação primeiro.")
            sys.exit(1)
        filepaths = [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if f.endswith(".csv")
        ]

    plot_comparison(filepaths, args.output)


if __name__ == "__main__":
    main()
