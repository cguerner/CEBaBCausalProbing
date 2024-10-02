# column names
OVERALL_LABEL = 'review_majority'
DESCRIPTION = 'description'
TREATMENTS = ['food', 'ambiance', 'service', 'noise']
NO_MAJORITY = 'no majority'
ID_COL = 'id'

# possible concept values
POSITIVE = 'Positive'
NEGATIVE = 'Negative'
UNKNOWN = 'unknown'

# task names
OPENTABLE_BINARY = 'opentable_binary'
OPENTABLE_TERNARY = 'opentable_ternary'
OPENTABLE_5_WAY = 'opentable_5_way'

# seeds
SEEDS_ELDAR = list(range(42, 47))
SEEDS_ZEN = [42, 66, 77, 88, 99]
SEEDS_ELDAR2ZEN = {e: z for e, z in zip(SEEDS_ELDAR, SEEDS_ZEN)}
SEEDS_ZEN2ELDAR = {v: k for k, v in SEEDS_ELDAR2ZEN.items()}

# pretrained model checkpoints
BERT = 'bert-base-uncased'
T5 = 't5-base'
GPT2 = 'gpt2'
ROBERTA = 'roberta-base'
LSTM = 'lstm'

MODELS = [BERT, T5, GPT2, ROBERTA, LSTM]

# misc
CEBAB = 'CEBaB'

# auth_token_path
AUTH_TOKEN_PATH = "/cluster/work/cotterell/cguerner/CEBaB/auth_token.txt"

def load_auth_token():
    with open(AUTH_TOKEN_PATH) as f:
        lines = f.readlines()

    return lines[0]