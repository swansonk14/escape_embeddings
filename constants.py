"""Contains constants used in the Python files in this repo."""
from typing import Literal

AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_ALPHABET_SET = set(AA_ALPHABET)
HEAVY_CHAIN = 'heavy'
LIGHT_CHAIN = 'light'
ANTIBODY_CHAINS = [HEAVY_CHAIN, LIGHT_CHAIN]
PROTEIN_LINKER = 'GGS' * 7  # Linker amino acids for combining multiple proteins into a single sequence

# Constants for data from Cao et al. https://www.nature.com/articles/s41586-021-04385-3
ANTIBODY_COLUMN = 'condition'
ANTIBODY_NAME_COLUMN = 'name'
EPITOPE_GROUP_COLUMN = 'epitope group'
ESCAPE_COLUMN = 'mut_escape'
HEAVY_CHAIN_COLUMN = 'Hchain'
LIGHT_CHAIN_COLUMN = 'Lchain'
MUTANT_COLUMN = 'mutation'
SITE_COLUMN = 'site'
WILDTYPE_COLUMN = 'wildtype'
ANTIBODY_CONDITION_TO_NAME = {
    'COV2-2130': 'AZD1061',
    'COV2-2196': 'AZD8895',
    'S309': 'VIR-7831'
}
ANTIBODY_NAME_TO_CONDITION = {name: condition for condition, name in ANTIBODY_CONDITION_TO_NAME.items()}

# SARS-CoV-2 RDB is extracted from Cao et al. using extract_rbd.py
RBD_SITE_TO_AA = {
    331: 'N', 332: 'I', 333: 'T', 334: 'N', 335: 'L', 336: 'C', 337: 'P', 338: 'F', 339: 'G', 340: 'E',
    341: 'V', 342: 'F', 343: 'N', 344: 'A', 345: 'T', 346: 'R', 347: 'F', 348: 'A', 349: 'S', 350: 'V',
    351: 'Y', 352: 'A', 353: 'W', 354: 'N', 355: 'R', 356: 'K', 357: 'R', 358: 'I', 359: 'S', 360: 'N',
    361: 'C', 362: 'V', 363: 'A', 364: 'D', 365: 'Y', 366: 'S', 367: 'V', 368: 'L', 369: 'Y', 370: 'N',
    371: 'S', 372: 'A', 373: 'S', 374: 'F', 375: 'S', 376: 'T', 377: 'F', 378: 'K', 379: 'C', 380: 'Y',
    381: 'G', 382: 'V', 383: 'S', 384: 'P', 385: 'T', 386: 'K', 387: 'L', 388: 'N', 389: 'D', 390: 'L',
    391: 'C', 392: 'F', 393: 'T', 394: 'N', 395: 'V', 396: 'Y', 397: 'A', 398: 'D', 399: 'S', 400: 'F',
    401: 'V', 402: 'I', 403: 'R', 404: 'G', 405: 'D', 406: 'E', 407: 'V', 408: 'R', 409: 'Q', 410: 'I',
    411: 'A', 412: 'P', 413: 'G', 414: 'Q', 415: 'T', 416: 'G', 417: 'K', 418: 'I', 419: 'A', 420: 'D',
    421: 'Y', 422: 'N', 423: 'Y', 424: 'K', 425: 'L', 426: 'P', 427: 'D', 428: 'D', 429: 'F', 430: 'T',
    431: 'G', 432: 'C', 433: 'V', 434: 'I', 435: 'A', 436: 'W', 437: 'N', 438: 'S', 439: 'N', 440: 'N',
    441: 'L', 442: 'D', 443: 'S', 444: 'K', 445: 'V', 446: 'G', 447: 'G', 448: 'N', 449: 'Y', 450: 'N',
    451: 'Y', 452: 'L', 453: 'Y', 454: 'R', 455: 'L', 456: 'F', 457: 'R', 458: 'K', 459: 'S', 460: 'N',
    461: 'L', 462: 'K', 463: 'P', 464: 'F', 465: 'E', 466: 'R', 467: 'D', 468: 'I', 469: 'S', 470: 'T',
    471: 'E', 472: 'I', 473: 'Y', 474: 'Q', 475: 'A', 476: 'G', 477: 'S', 478: 'T', 479: 'P', 480: 'C',
    481: 'N', 482: 'G', 483: 'V', 484: 'E', 485: 'G', 486: 'F', 487: 'N', 488: 'C', 489: 'Y', 490: 'F',
    491: 'P', 492: 'L', 493: 'Q', 494: 'S', 495: 'Y', 496: 'G', 497: 'F', 498: 'Q', 499: 'P', 500: 'T',
    501: 'N', 502: 'G', 503: 'V', 504: 'G', 505: 'Y', 506: 'Q', 507: 'P', 508: 'Y', 509: 'R', 510: 'V',
    511: 'V', 512: 'V', 513: 'L', 514: 'S', 515: 'F', 516: 'E', 517: 'L', 518: 'L', 519: 'H', 520: 'A',
    521: 'P', 522: 'A', 523: 'T', 524: 'V', 525: 'C', 526: 'G', 527: 'P', 528: 'K', 529: 'K', 530: 'S',
    531: 'T'
}
RBD_SITES = sorted(RBD_SITE_TO_AA)
RBD_START_SITE = RBD_SITES[0]  # inclusive
RBD_END_SITE = RBD_SITES[-1]  # inclusive
RBD_SEQUENCE = ''.join(RBD_SITE_TO_AA[site] for site in RBD_SITES)

# Literal types
MODEL_GRANULARITY_OPTIONS = Literal['per-antibody', 'cross-antibody']
MODEL_TYPE_OPTIONS = Literal['mutation', 'site', 'likelihood', 'embedding']
TASK_TYPE_OPTIONS = Literal['classification', 'regression']
SPLIT_TYPE_OPTIONS = Literal['mutation', 'site', 'antibody', 'antibody_group']
ANTIBODY_GROUP_METHOD_OPTIONS = Literal['sequence', 'embedding', 'escape']
EMBEDDING_GRANULARITY_OPTIONS = Literal['sequence', 'residue']
ANTIGEN_EMBEDDING_TYPE_OPTIONS = Literal['mutant', 'difference']
ANTIBODY_EMBEDDING_TYPE_OPTIONS = Literal['concatenation', 'attention']

# Model constants
DEFAULT_BATCH_SIZE = 100
DEFAULT_HIDDEN_LAYER_DIMS = (100, 100, 100)
DEFAULT_NUM_EPOCHS_CROSS_ANTIBODY = 1
DEFAULT_NUM_EPOCHS_PER_ANTIBODY = 50
