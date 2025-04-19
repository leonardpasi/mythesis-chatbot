import click
from trulens.dashboard import run_dashboard

from src.mythesis_chatbot.evaluation import get_tru_session


@click.command(context_settings={"show_default": True})
@click.option(
    "--db",
    "database",
    type=click.Choice(["prod", "dev"], case_sensitive=False),
    default="prod",
)
def main(database: str):
    tru = get_tru_session(database)
    run_dashboard(tru)


main()
