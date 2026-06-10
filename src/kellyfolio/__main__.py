"""CLI: python -m kellyfolio report --value 10000"""
import argparse

import yaml

from kellyfolio.data import get_prices
from kellyfolio.report import kelly_strategy, format_report


def main():
    parser = argparse.ArgumentParser(prog="kellyfolio")
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="print today's suggested allocation")
    report.add_argument("--value", type=float, default=10_000,
                        help="portfolio value in dollars")
    report.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    prices = get_prices(config["tickers"], config["start_date"])
    weights = kelly_strategy(config)(prices)
    if weights is None:
        raise SystemExit("Not enough price history for the configured lookback.")
    print(format_report(weights, args.value))


if __name__ == "__main__":
    main()
