"""Council of Achrona: een mini-Council met Raadsleden en een Koning.

Start met `uv run council`. Schrijft verdict.json en trace.json in de root van de repo.
De Koning leidt het debat in `kies_volgende()`: hij beslist wie spreekt, met welke vraag en waarom.
"""
import asyncio
import json
import os
import sys
import uuid
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from openai import AsyncOpenAI

import raadsleden

ROOT = Path(__file__).parent
KENNISBRONNEN = ROOT / "kennisbronnen"
ROLLEN = ("archmage", "engineer", "guardian_keeper", "historian", "oracle", "beast_master")
BEURTENBUDGET = 10  # spelregel: maximaal 10 Raadslid-beurten (berichten van Raadsleden) per run


async def kies_volgende(beurten, leden):
    """De Koning leidt het debat: hij beslist wie spreekt. De kit roept deze functie op vóór elke beurt.
    (In AI-termen heet dit deel een orchestrator.)

    Je krijgt:
      beurten  het debat tot nu toe, dezelfde lijst als in trace.json. Een bericht van een
               Raadslid ziet er zo uit: {"soort": "bericht", "spreker": "engineer", "tekst": "..."}
      leden    de Raadsleden uit raadsleden/, elk met .ROL, .NAAM en .PERSPECTIEF

    Je geeft terug:
      (rol, vraag, reden)   wie nu spreekt, welke vraag die krijgt en waarom je hem kiest
      (None, None, reden)   het debat stopt, met de reden waarom
    De reden komt in de trace en in de viewer. De jury leest ze: maak ze concreet.

    TODO voor jullie team:
    Nu praat iedereen om de beurt, in een vaste volgorde, ook als dat niets toevoegt. Er zijn maar
    10 Raadslid-beurten, dus zo verspil je beurten. Laat de Koning in deze functie naar het debat
    kijken en elke beurt bewust kiezen. Vragen om mee te starten:
      - Wie heeft nog niet gesproken, en wiens kennis ontbreekt nog?
      - Iemand verwijst naar een andere Rol of doet een bewering: wie kan dat controleren?
      - Welke delen van de oorzaak (ingreep, mechanisme, verloop) zijn nog onbekend?
      - Weet de Council genoeg? Stop dan zelf, vóór het budget op is.
    Twee manieren (of een mix):
      - regels in code, bv. "wie genoemd wordt, moet het toetsen";
      - een agent: de Koning als LLM die het debat leest en (rol, vraag, reden) teruggeeft.
        Meestal sneller gebouwd en slimmer dan regels: zie "Regels of een agent?" in de README.
    """
    "U bent de koning en leidt een debat. U krijgt een lijst van beurten en een lijst van leden. Op basis van het debat tot nu toe, kies wie nu spreekt, welke vraag die krijgt en waarom. Geef terug: (rol, vraag, reden). Als er genoeg informatie is voor een verdict, stop je het debat. Geef dan None als rol."

    opties = {"reasoning_effort": os.environ["REASONING"]} if os.getenv("REASONING") else None
    instructies = (
        "U bent de koning en leidt een debat. U krijgt een lijst van beurten en een lijst van leden. "
        "Kies wie nu spreekt, welke vraag die krijgt en waarom. "
        "Als er genoeg informatie is voor een verdict, stop je het debat en geef je rol null. "
        "Gebruik geen file namen in uw berichten, vraag het lid gewoon over zijn informatie. Bron ID's mogen wel genoemd worden. "
        'Antwoord alleen met JSON, bv. {"rol": "engineer", "vraag": "...", "reden": "..."}'
    )
    bericht = (
        "Beurten:\n" + json.dumps(beurten, indent=2, ensure_ascii=False)
        + "\n\nLeden:\n" + json.dumps([{"ROL": l.ROL, "NAAM": l.NAAM, "PERSPECTIEF": l.PERSPECTIEF} for l in leden], indent=2, ensure_ascii=False)
    )
    koning = Agent(client("MODEL_GROOT"), instructies, name="koning")
    r = await koning.run(bericht, options=opties)
    tekst = r.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    keuze = json.loads(tekst)
    return keuze["rol"], keuze["vraag"], keuze["reden"]


def controleer(leden):
    """Duidelijke melding in plaats van een traceback bij de meest voorkomende fouten."""
    mist = [v for v in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_KLEIN", "MODEL_GROOT") if not os.getenv(v)]
    if mist:
        sys.exit(f"In .env ontbreekt: {', '.join(mist)}. Kopieer .env.example naar .env en vul het in, of zie 'Vooraf installeren' in de README.")
    for lid in leden:
        if lid.ROL not in ROLLEN:
            sys.exit(f"raadsleden/: ROL = {lid.ROL!r} is ongeldig. Gebruik de mapnaam van de Rol: {', '.join(ROLLEN)}.")
    dubbel = {l.ROL for l in leden if [x.ROL for x in leden].count(l.ROL) > 1}
    if dubbel:
        sys.exit(f"Eén Raadslid per Rol: {', '.join(dubbel)} komt meer dan eens voor in raadsleden/.")


def client(model_env):
    key = os.environ["OPENAI_API_KEY"]
    return OpenAIChatCompletionClient(
        model=os.environ[model_env],
        async_client=AsyncOpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=key,
            max_retries=8,  # bij een rate limit (429) wacht de client en probeert opnieuw
        ),
    )


def kennisbron_bestanden(rol):
    map_ = KENNISBRONNEN / rol
    return sorted(p for p in map_.glob("*") if p.suffix in (".md", ".csv")) if map_.is_dir() else []


def instructies(lid, bestanden, mcp):
    tekst = (
        f"Je bent de {lid.NAAM}, een Raadslid in de Council of Achrona. {lid.PERSPECTIEF}\n"
        "Je debatteert met de andere Raadsleden over de oorzaak van de Desync. "
        "Baseer je alleen op je eigen Kennisbron en noem bij elke bewering de record-ID's precies zoals ze in je Kennisbron staan. Verzin nooit een ID. "
        "Zeg het eerlijk als je iets niet weet."
    )
    if mcp:
        tekst += "\nLees of doorzoek je Kennisbron met je tools voor je spreekt."
    elif bestanden:
        tekst += "\n\nJe Kennisbron:\n\n" + "\n\n".join(f"## {p.name}\n\n{p.read_text(encoding='utf-8')}" for p in bestanden)
    else:
        tekst += "\nJe hebt nog geen Kennisbron: zeg dat je niets kan bewijzen."
    return tekst


def tel(verbruik, model, usage):
    if not usage:
        return
    v = verbruik.setdefault(model, {"beurten": 0, "invoer": 0, "gecached": 0, "uitvoer": 0})
    v["beurten"] += 1  # één beurt = één agent-run (met MCP soms meerdere model-requests)
    v["invoer"] += usage.get("input_token_count") or 0
    v["gecached"] += usage.get("cache_read_input_token_count") or 0
    v["uitvoer"] += usage.get("output_token_count") or 0


def toolcalls(response):
    for msg in response.messages:
        for c in msg.contents:
            if c.type == "function_call":
                args = c.arguments
                yield c.name, json.loads(args) if isinstance(args, str) and args else (args or {})


async def run():
    load_dotenv(ROOT / ".env")
    (ROOT / "verdict.json").unlink(missing_ok=True)
    (ROOT / "trace.json").unlink(missing_ok=True)
    opties = {"reasoning_effort": os.environ["REASONING"]} if os.getenv("REASONING") else None  # minder denkwerk = minder tokens
    run_id = f"{datetime.now().astimezone().isoformat(timespec='seconds')}_{uuid.uuid4().hex[:8]}"
    leden = raadsleden.alle()
    controleer(leden)
    beurten, verbruik = [], {}
    trace = {"run_id": run_id, "status": "bezig", "raadsleden": [l.ROL for l in leden], "beurten": beurten, "geschillen": [], "verbruik": verbruik}

    def bewaar(naam, data):
        (ROOT / naam).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def beurt(ronde, spreker, soort, **velden):
        beurten.append({"nr": len(beurten) + 1, "ronde": ronde, "spreker": spreker, "soort": soort, **velden})
        bewaar("trace.json", trace)  # na elke beurt: de viewer (`uv run viewer`) toont het debat live

    klein = client("MODEL_KLEIN")

    async with AsyncExitStack() as stack:
        agents = []
        for lid in leden:
            mcp_url = os.getenv(f"MCP_{lid.ROL.upper()}", "").strip()
            bestanden = [] if mcp_url else kennisbron_bestanden(lid.ROL)
            tools = []
            if mcp_url:
                try:
                    tools.append(await stack.enter_async_context(MCPStreamableHTTPTool(name=f"kennisbron_{lid.ROL}", url=mcp_url)))
                except Exception as e:
                    sys.exit(f"MCP voor {lid.NAAM} ({mcp_url}) is onbereikbaar: {e}\n"
                             f"Is de server al open (vanaf 10:00)? Klopt het pad /{lid.ROL}/mcp? "
                             f"Of laat MCP_{lid.ROL.upper()} leeg om de bestanden uit kennisbronnen/{lid.ROL}/ te gebruiken.")
            elif not bestanden:
                print(f"LET OP: {lid.NAAM} heeft geen Kennisbron (MCP_{lid.ROL.upper()} is leeg en kennisbronnen/{lid.ROL}/ ook).")
            for p in bestanden:  # fallback zonder MCP: welke bestanden gingen in de prompt
                beurt(0, lid.ROL, "bestand", bestand=f"{lid.ROL}/{p.name}")
            agents.append((lid, mcp_url, Agent(klein, instructies(lid, bestanden, mcp_url), name=lid.ROL, tools=tools)))

        (ROOT / "verdict.json").unlink(missing_ok=True)  # setup gelukt: geen Verdict van een vorige run laten staan
        bewaar("trace.json", trace)
        per_rol = {lid.ROL: (lid, mcp_url, agent) for lid, mcp_url, agent in agents}
        debat, ronde = [], 1
        while True:
            gesproken = sum(1 for b in beurten if b["soort"] == "bericht" and b["spreker"] in per_rol)
            if gesproken >= BEURTENBUDGET:
                rol, vraag, reden = None, None, f"Beurtenbudget op ({BEURTENBUDGET} Raadslid-beurten)"
            else:
                rol, vraag, reden = await kies_volgende(beurten, leden)
            if rol is not None and rol not in per_rol:
                sys.exit(f"kies_volgende() koos {rol!r}, maar die Rol zit niet in raadsleden/.")
            if rol is not None:
                ronde = gesproken // len(leden) + 1
            # Elke keuze van de Koning staat in de trace: aan wie, welke vraag en waarom.
            beurt(ronde, "koning", "routering", aan=rol, vraag=vraag, reden=reden)
            print(f"\n[Koning] {'stopt' if rol is None else '→ ' + per_rol[rol][0].NAAM}: {reden}")
            if rol is None:
                break
            lid, mcp_url, agent = per_rol[rol]
            # Wie ziet wat? Hier ziet elk Raadslid het hele debat (broadcast). Ook dat is een keuze.
            prompt = "Debat tot nu toe:\n\n" + ("\n\n".join(debat) or "(nog niets)") + f"\n\nVraag aan jou: {vraag}"
            trace["aan_het_woord"] = rol
            bewaar("trace.json", trace)
            r = await agent.run(prompt, options=opties)
            tel(verbruik, "klein", r.usage_details)
            for tool, args in toolcalls(r):
                beurt(ronde, rol, "toolcall", tool=tool, argumenten=args, pad=mcp_url)
            beurt(ronde, rol, "bericht", tekst=r.text)
            debat.append(f"{lid.NAAM}: {r.text}")
            print(f"\n[ronde {ronde}] {lid.NAAM}:\n{r.text}")

    # De Koning heeft geen Rol en geen Kennisbron: hij kent alleen wat de Raadsleden inbrachten.
    trace["aan_het_woord"] = "koning"
    bewaar("trace.json", trace)
    koning = Agent(client("MODEL_GROOT"), "Je bent de Koning van de Council of Achrona. Je hebt geen Kennisbron.", name="koning")
    r = await koning.run(
        options=opties,
        messages="Debat:\n\n" + "\n\n".join(debat)
        + "\n\nVat samen: waarover zijn de Raadsleden het eens, waarover niet, en welk bewijs (record-ID's) noemden ze?",
    )
    tel(verbruik, "groot", r.usage_details)
    beurt(ronde + 1, "koning", "bericht", tekst=r.text)
    print(f"\n[Koning]:\n{r.text}")

        # De Koning schrijft het Verdict op basis van het debat en de samenvatting
    schema = (ROOT / "schema" / "verdict.json").read_text(encoding="utf-8")
    verdict_prompt = (
        "Debat:\n\n" + "\n\n".join(debat)
        + "\n\nSamenvatting:\n" + r.text
        + "\n\nSchrijf nu het Verdict van de Council als JSON volgens dit schema:\n" + schema
        + "\n\nGebruik bij bewijs alleen record-ID's die de Raadsleden echt noemden. Antwoord alleen met JSON."
    )
    r = await koning.run(verdict_prompt, options=opties)
    tel(verbruik, "groot", r.usage_details)
    tekst = r.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    verdict = json.loads(tekst)
    verdict["run_id"] = run_id
    trace["status"], trace["aan_het_woord"] = "klaar", None
    bewaar("trace.json", trace)
    bewaar("verdict.json", verdict)

    print(f"\nKlaar: verdict.json en trace.json geschreven (run_id {run_id}).")
    for model, v in verbruik.items():
        print(f"Verbruik {model}: {v['beurten']} beurten, {v['invoer']} tokens in ({v['gecached']} gecached), {v['uitvoer']} tokens uit")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(run())


if __name__ == "__main__":
    main()
