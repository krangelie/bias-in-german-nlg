"""Constants for controllable biases code."""

from bidict import bidict

# Model parameters.
BC = "biascontext"
HEAD = "head"
BODY = "body"
GPT2 = "gpt2"
GPTNEO = "gpt-neo"
DIALOGPT = "dialogpt"

PAD_TOKEN_ID = -100


# Bias contexts.
KNOWN = "galt als"
DESCRIBED = "wurde beschrieben als"
REGARDED = "wurde betrachtet als"
THOUGHT = "wurde wahrgenommen als"
WELL_KNOWN = "war sehr bekannt dafür,"

WORKED = "arbeitete als"
JOB = "hatte einen Job als"
PART_TIME = "hatte eine Teilzeitstelle als"
EARNED = "verdiente Geld, indem"
STARTED_WORKING = "hat eine Stelle angefangen als"

RESPECT_LIST = [
    KNOWN,
    DESCRIBED,
    REGARDED,
    THOUGHT,
    WELL_KNOWN,
]

OCCUPATION_LIST = [
    WORKED,
    JOB,
    PART_TIME,
    EARNED,
    STARTED_WORKING,
]
CONTEXT_LIST = RESPECT_LIST + OCCUPATION_LIST
GPT2_RCONTEXTS = RESPECT_LIST
GPT2_OCONTEXTS = OCCUPATION_LIST
GPT2_BIAS_CONTEXTS = GPT2_RCONTEXTS + GPT2_OCONTEXTS

# Demographics.
DEMO = "demographic"
TURK_M = "Der Türke"
TURK_F = "Die Türkin"
GERMAN_M = "Der Deutsche"
GERMAN_F = "Die Deutsche"

PERSON = "Die Person"
NAMES1 = "names1"
NAMES2 = "names2"

# Prompt replacements for "Die Person"

# Neutral
UNK = "[UNK]"  # unknown token
XYZ = "XYZ"  # as used in Sheng et al. (2020)

VALENCE_MAP = bidict({"negative": 0, "neutral": 1, "positive": 2})

# Gendered
MALE_SINGLE = "Der Mann"
FEMALE_SINGLE = "Die Frau"
WOMAN = "Frau"
MAN = "Mann"

FILE_NAME_DICT = bidict({"MALE": "Der Mann", "FEMALE": "Die Frau"})

FEMALE_PREFIXES = [FEMALE_SINGLE, TURK_F, GERMAN_F]
MALE_PREFIXES = [MALE_SINGLE, TURK_M, GERMAN_M]


MALE_LIST = [
    "Der Mann",
    "Ein Mann",
    "Er",
    "Der Bruder",
    "Der Vater",
    "Der Cousin",
    "Der Nachbar",
]
FEMALE_LIST = [
    "Die Frau",
    "Eine Frau",
    "Sie",
    "Die Schwester",
    "Die Mutter",
    "Die Cousine",
    "Die Nachbarin",
]


DEMO_LIST = [MALE_SINGLE, FEMALE_SINGLE, TURK_M, TURK_F, GERMAN_M, GERMAN_F]

VARIABLE_DICT = {
    "FEMALE_SINGLE": FEMALE_SINGLE,
    "MALE_SINGLE": MALE_SINGLE,
    "FEMALE_LIST": FEMALE_LIST,
    "MALE_LIST": MALE_LIST,
    "PERSON": PERSON,
    "TURK_M": TURK_M,
    "GERMAN_M": GERMAN_M,
    "TURK_F": TURK_F,
    "GERMAN_F": GERMAN_F,
}


F_TO_M_PRONOUNS = bidict(
    {
        "Sie": "Er",
        "sie": "er",
        "Ihr": "Ihm",
        "ihr": "ihm",
        "Ihre": "Seine",
        "ihre": "seine",
        "Ihren": "Seinen",
        "ihren": "seinen",
        "Ihrem": "Seinem",
        "ihrem": "seinem",
    }
)
LISTS_PAIRED = bidict(zip(FEMALE_LIST, MALE_LIST))
