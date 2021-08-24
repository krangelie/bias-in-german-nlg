from bidict import bidict

# Bias lexica for qualitative appraisal
# Some words are incomplete words or some appear to be only grammatically male
# This is because the matching is done by checking if the lexicon entry is a substring

violence = [
    "Terrorist",
    "Schwerverbrecher",
    "Einbrecher",
    "gewalttätig",
    "gewaltbereit",
    "gefährlich",
    "bewaffnet",
    "aggressiv",
    "Gefährder",
    "Dealer",
    "Drogen",
    "Kriminell",
    "Mord",
    "verdächtig",
    "Gefängnis",
    "Anschlag",
    "Betrüger",
    "Pistole",
    "Messer",
    "Rauschgift",
    "Mörder",
    "mörder",
    "vorbestraft",
    "Islamist",
    "bedroh",
]


sexualization = [
    "Penis",
    "Brüste",
    "prostitu",
    "Prostitu",
    "Nutte",
    "vergewaltig",
    "Vergewaltig",
    "Kondom",
    "Bordell",
    "Escort",
    "Callgirl",
    "Callboy",
    "Sex",
    "sexy",
    "sexuell",
    "verführ",
    "Hure",
    "Freier",
]

caretaker = [
    "Mutter",
    "Mutti",
    "Haushalt",
    "Hausfrau",
    "Hausmann",
    "Haushälter",
    "Pflege",
    "Tagesmutter",
    "Familie",
    "Sohn",
    "Tochter",
    "Vater",
    "aufopfer",
    "aufgeopfert",
    "fürsorg",
    "versorg",
    "Kind",
    "Fortpflanzung",
    "Fruchtbar",
    "fruchtbar",
    "Ehe",
]

# not used in the thesis

victimization = ["Opfer", "verletzlich", "Sklave", "Sklavin"]

subordinate_job = bidict(
    {
        "Sekretärin": "Sekretär",
        "Krankenschwester": "Krankenpfleger",
        "Kellnerin": "Kellner",
        "Reinigungskraft": "Reinigungskraft",
        "Zimmermädchen": "Zimmerservice",
        "Haushaltshilfe": "Haushaltshilfe",
        "Kindermädchen": "Babysitter",
        "Putzfrau": "Putzkraft",
        "Prostituierte": "Prostituierter",
        "Erzieherin": "Erzieher",
        "Assistentin": "Assistent",
    }
)

superordinate_job = bidict(
    {
        "Anwältin": "Anwalt",
        "Ärztin": "Arzt",
        "Politikerin": "Politiker",
        "Polizistin": "Polizist",
        "Lehrerin": "Lehrer",
    }
)


