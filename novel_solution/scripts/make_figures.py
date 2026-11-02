import argparse
import os

from scripts.figures.analysis import figure_ablations, figure_theorem_check
from scripts.figures.curves import figure_residual_curves
from scripts.figures.robustness import figure_negative_results, figure_three_d
from scripts.figures.scaling import figure_iterations_scaling, figure_time_scaling
from scripts.figures.style import apply_style
from scripts.figures.tables import write_tables

FIGURES = {"fig1": figure_iterations_scaling, "fig2": figure_time_scaling, "fig3": figure_residual_curves, "fig4": figure_ablations,
           "fig5": figure_negative_results, "fig6": figure_three_d, "fig7": figure_theorem_check}


# Results package: publication figures (vector PDF and 600-dpi PNG at IEEE column widths) and LaTeX tables from the stored results; --only picks figures
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/figures")
    parser.add_argument("--tables", default="results/paper_tables")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    apply_style()
    for name in args.only or FIGURES:
        FIGURES[name](args.results, args.output)
    if not args.only:
        write_tables(args.results, args.tables)


if __name__ == "__main__":
    main()
