import os
import sys
from dotenv import load_dotenv, find_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pprint import pprint
from api.agent_api import AgentAPI


def main() -> None:
    load_dotenv(find_dotenv())

    ag = AgentAPI()
    ag.clear()
    pprint(ag.query("Olá, tudo bem?"))
    pprint(ag.query("O que o time de IAG faz na tech4ai?"))
    # ag.query(
    #     "Gostaria de criar um evento de reunião para hoje às 15h. Poderia me ajudar?"
    # )
    # ag.query("Você lembra meu nome?")

    # pprint(ag.query("O que é GitHub?"))
    # pprint(ag.query("O que é commit e pull no git?"))


if __name__ == "__main__":
    main()
